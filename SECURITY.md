# Security Policy

This is a portfolio/demo project. The security practices here reflect what's appropriate for a personal project while demonstrating awareness of production requirements.

## What This Project Does Right

- **Secrets are never committed.** `.env`, `secrets.toml`, and all credential files are in `.gitignore`. The CI pipeline includes an automated check that fails the build if secrets are accidentally tracked.
- **API keys are scoped.** FRED, NYC OpenData, and Anthropic keys are read-only with no write access to external systems.
- **SEC rate limits are enforced in code.** The SEC fetcher enforces a 10 req/sec limit to comply with SEC Edgar policy.
- **API usage is logged.** All Anthropic API calls are logged with token counts and costs for monitoring.
- **Chatbot access is password-gated.** The SEC Chatbot page requires a password before allowing API calls.
- **Dependencies are pinned.** `requirements.txt` uses exact versions (`==`) to prevent supply chain drift.
- **Linting and CI.** Ruff linting and secret-scanning run on every push via GitHub Actions.

## Known Limitations

These are intentional trade-offs for a portfolio project and would be addressed in production:

| Area | Current State | Production Approach |
|------|--------------|-------------------|
| Secrets storage | `.env` / `secrets.toml` (gitignored) | AWS Secrets Manager, Azure Key Vault, or GCP Secret Manager |
| Authentication | Single password gate on chatbot | OAuth 2.0 / SSO with session management |
| Logging | Local file (`logs/api_usage.log`) | Centralized logging (Datadog, CloudWatch, Sentry) |
| Error details | Stack traces visible in dev | `showErrorDetails = false` in production config |
| HTTPS | Handled by Streamlit Cloud | TLS termination at load balancer |
| Rate limiting | SEC-only (code-enforced) | Application-level rate limiting for all endpoints |
| Input validation | Basic type checking | Parameterized queries, input sanitization |

## Dependencies

This project uses third-party packages from PyPI. Dependency security is managed through:

- Pinned versions in `requirements.txt`
- GitHub Dependabot alerts (enabled on the repository)
- Periodic manual review of dependency updates

## CI/CD Security

- GitHub Actions secrets are used for any credentials needed in CI (none currently required for lint-only pipeline)
- Branch protection rules can be enabled on `main` to require PR reviews and passing checks
- The lint workflow includes a secrets-in-code scan that blocks commits containing `.env` or `secrets.toml`
