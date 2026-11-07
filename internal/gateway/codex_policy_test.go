package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

const policyPassthroughError = `{"type":"invalid_request_error","code":"misalignment_policy_violation","message":"[ObjectParam] [input[0].internal_chat_message_metadata_passthrough.content_token_count] [unknown_parameter]","misalignment":{"detailed_explanation":"Synthetic review text.","review_target":"synthetic-review-target"}}`
const policyPassthroughItem = `{"type":"message","role":"user","content":[{"type":"input_text","text":"Synthetic conversation"}],"internal_chat_message_metadata_passthrough":{"content_token_count":7}}`
const policyPassthroughPath = "input.0." + passthroughKey + ".content_token_count"
const policyBadRequestFrame = `{"type":"error","status":400,"error":` + policyPassthroughError + `}`

func TestCodexCompactionPolicyError(t *testing.T) {
	const created = `{"type":"response.created","response":{"id":"resp_probe"}}`
	const policy = `{"type":"invalid_request_error","code":"misalignment_policy_violation","message":"Synthetic precaution","misalignment":{"detailed_explanation":"Synthetic review text.","review_target":"synthetic-review-target"}}`
	const wrapped = `{"type":"error","status":403,"error":` + policy + `}`
	const failed = `{"type":"response.failed","response":{"id":"resp_probe","status":"failed","error":` + policy + `}}`
	const compacted = `{"type":"response.output_item.done","output_index":0,"item":{"type":"compaction","encrypted_content":"synthetic-compaction"}}`
	const busy = `{"type":"error","status":503,"error":{"code":"server_is_overloaded","message":"Server is overloaded"}}`
	const completed = `{"type":"response.completed","response":{"id":"resp_probe","status":"completed"}}`

	for _, tc := range []struct {
		name   string
		frames []string
		status int
		event  string
	}{
		{"success", []string{created, compacted, completed}, http.StatusOK, "response.completed"},
		{"native-failed", []string{created, failed}, http.StatusOK, "response.failed"},
		{"early-400", []string{policyBadRequestFrame}, http.StatusBadRequest, ""},
		{"early-wrapped", []string{wrapped}, http.StatusForbidden, ""},
		{"late-wrapped", []string{created, wrapped}, http.StatusOK, "response.failed"},
		{"early-after-keepalive", []string{wrapped}, http.StatusOK, "response.failed"},
		{"early-eof", []string{created}, http.StatusOK, "response.created"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			socketOn(t)
			var attempt atomic.Int32
			wantAttempts := 1
			if tc.name == "early-after-keepalive" {
				wantAttempts = 2
				was := keepHeldAfter
				keepHeldAfter = 0
				t.Cleanup(func() { keepHeldAfter = was })
			}
			f := &fakeSocketBackend{frames: func(turn socketTurn) []string {
				if !gjson.GetBytes(turn.msg, "input.#(type==compaction_trigger)").Exists() {
					t.Error("upstream compaction trigger missing")
				}
				if turn.header.Get("Authorization") != "Bearer synthetic-codex-client" {
					t.Error("upstream caller credential changed")
				}
				if tc.name == "early-400" && attempt.Load() == 0 && !gjson.GetBytes(turn.msg, policyPassthroughPath).Exists() {
					t.Error("synthetic passthrough field never reached the upstream")
				}
				if attempt.Add(1) == 1 && tc.name == "early-after-keepalive" {
					return []string{busy}
				}
				return tc.frames
			}}
			chatgpt(t, f.ServeHTTP)
			rec := httptest.NewRecorder()
			req := httptest.NewRequest("POST", CodexPath+"/responses", strings.NewReader(
				`{"model":"gpt-6.1-sol","stream":true,"store":false,"input":[`+policyPassthroughItem+`,{"type":"compaction_trigger"}]}`))
			req.Header.Set("Authorization", "Bearer synthetic-codex-client")
			req.Header.Set("ChatGPT-Account-ID", "synthetic-account")
			req.Header.Set("Accept", "text/event-stream")
			req.Header.Set("X-Codex-Turn-Metadata", `{"compaction":{"trigger":"manual","implementation":"responses_compaction_v2","phase":"standalone_turn"}}`)
			New().Handler().ServeHTTP(rec, req)
			body := rec.Body.String()
			if out := os.Getenv("MAGPIE_COMPACTION_PROBE_OUT"); out != "" {
				if err := os.WriteFile(filepath.Join(out, tc.name+".txt"), []byte(body), 0600); err != nil {
					t.Fatal(err)
				}
			}
			posts, sockets := f.counts()
			if posts != 0 || sockets != wantAttempts {
				t.Fatalf("want %d WS attempts with the original identity, got %d HTTP / %d WS", wantAttempts, posts, sockets)
			}
			if rec.Code != tc.status {
				t.Fatalf("status %d, want %d", rec.Code, tc.status)
			}
			if tc.event != "" && !strings.Contains(body, "event: "+tc.event+"\n") {
				t.Fatalf("expected Codex-readable %s; got %s", tc.event, body)
			}
			if tc.name == "early-after-keepalive" && !strings.HasPrefix(body, ": keepalive\n") {
				t.Fatal("expected the downstream stream to start before the policy error")
			}
			if tc.name != "success" && tc.name != "early-eof" {
				wantPolicy := policy
				if tc.name == "early-400" {
					wantPolicy = policyPassthroughError
				}
				if !strings.Contains(body, wantPolicy) || strings.Contains(body, "event: response.completed\n") {
					t.Fatal("lost structured policy details or produced a success")
				}
			}
			if tc.name == "early-eof" && (strings.Contains(body, "event: response.completed\n") || strings.Contains(body, "misalignment_policy_violation")) {
				t.Fatal("proxy fabricated an upstream outcome")
			}
		})
	}
}

// A saved account's policy stop must reach Codex with the original review
// data even if its message matches a known request repair or another
// account is available.
func TestCodexManagedPolicyErrorEndsRequest(t *testing.T) {
	sealedPolicy := strings.Replace(policyPassthroughError,
		"[ObjectParam] [input[0].internal_chat_message_metadata_passthrough.content_token_count] [unknown_parameter]",
		"Encrypted content could not be decrypted", 1)
	for _, tc := range []struct {
		name, input, path, policy string
		status                    int
	}{
		{"passthrough", policyPassthroughItem, policyPassthroughPath, policyPassthroughError, http.StatusBadRequest},
		{"sealed history", policyPassthroughItem + `,{"type":"reasoning","id":"rs_synthetic","encrypted_content":"synthetic-reasoning","summary":[]}`,
			"input.1.encrypted_content", sealedPolicy, http.StatusForbidden},
	} {
		t.Run(tc.name, func(t *testing.T) {
			codexSignedIn(t, "spare@example.com")
			socketOn(t)
			var attempts atomic.Int32
			f := &fakeSocketBackend{frames: func(turn socketTurn) []string {
				if attempts.Add(1) == 1 {
					if !gjson.GetBytes(turn.msg, tc.path).Exists() {
						t.Error("synthetic history field never reached the upstream")
					}
					return []string{fmt.Sprintf(`{"type":"error","status":%d,"error":%s}`, tc.status, tc.policy)}
				}
				return wholeReply(turn)
			}}
			up := httptest.NewServer(f)
			t.Cleanup(up.Close)
			was := provider.CodexBase
			provider.CodexBase = up.URL + CodexPath
			t.Cleanup(func() { provider.CodexBase = was })

			code, body := codexPost(t, `{"model":"codex/gpt-5.5","stream":true,"input":[`+tc.input+`]}`)
			posts, sockets := f.counts()
			if code != tc.status || !strings.Contains(body, tc.policy) || posts != 0 || sockets != 1 {
				t.Fatalf("want the original policy error after one socket attempt, got HTTP %d, %d posts, %d sockets, body %s", code, posts, sockets, body)
			}
		})
	}
}
