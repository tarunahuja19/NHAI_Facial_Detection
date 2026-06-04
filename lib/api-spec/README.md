# Shared API Specification (`@workspace/api-spec`)

This package houses the OpenAPI contract and the code generation configuration that coordinates type-safety and contract consistency across the NHAI monorepo.

---

## 📂 Contents

* **`openapi.yaml`**: The source of truth API contract adhering to OpenAPI 3.1.0. It defines the schemas, path operations, request bodies, and responses for the API endpoints.
* **`orval.config.ts`**: The configuration file for [Orval](https://orval.dev/). It dictates how the API client (`@workspace/api-client-react`) and Zod schemas (`@workspace/api-zod`) are generated from `openapi.yaml`.

---

## 🛠️ Code Generation

When you make changes to `openapi.yaml`, run this command from the monorepo root to regenerate files in `@workspace/api-client-react` and `@workspace/api-zod`:

```bash
pnpm --filter @workspace/api-spec run codegen
```

### Generated Targets
1. **React Query Hooks**: Saved in `@workspace/api-client-react/src/generated/`. It uses a custom HTTP fetch wrapper (`custom-fetch.ts`) to request backend endpoints.
2. **Zod Validator Schemas**: Saved in `@workspace/api-zod/src/generated/`. It defines validation logic matching endpoint schema expectations, used by the server to validate payloads.
