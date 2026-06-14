# Security and Privacy

## Credentials

- Never commit `.env`, `.env.local`, model API keys, Shuiyuan User API Keys, authorization payloads, or JAccount credentials.
- The Web UI requests a Discourse User API Key with the `read` scope only.
- Credentials entered in the Web UI are sent only to the local same-origin Python server for the current request. The UI does not persist them in browser storage.
- The LLM endpoint is restricted to hosts listed in `SJTU_LLM_ALLOWED_BASE_HOSTS`. The default allows only `models.sjtu.edu.cn`.
- The Shuiyuan endpoint is restricted to hosts listed in `SHUIYUAN_ALLOWED_BASE_HOSTS`. The default allows only `shuiyuan.sjtu.edu.cn`.

## Community Content

- Shuiyuan searches and topic bodies are fetched on demand and are not committed to this repository.
- Do not publish private community content or authorization tokens.
- Respect Shuiyuan rate limits. The client waits for retryable rate-limit responses and uses a bounded total wait.
- Publishing this source code does not grant Shuiyuan access. Each user must complete Shuiyuan authentication and authorize their own read-only User API Key.
- Authentication is not a substitute for secret handling: a leaked or borrowed authorized key can still access content within its scope.
- Keep the service bound to `127.0.0.1` unless the deployment adds its own authentication, access controls, and credential isolation.

## Reporting

If you discover a vulnerability, report it privately to the repository maintainers rather than opening a public issue containing credentials or private content.
