# Shared API Validation Schemas (`@workspace/api-zod`)

This library provides TypeScript-compatible Zod validation schemas and type declarations representing the API request payloads, query parameters, and responses. 

---

## 📦 Features

* **Auto-generated Validators**: Compiled directly from `openapi.yaml` using [Orval](https://orval.dev/).
* **Consistent Data Verification**: Shared between the mobile app (to pre-validate data before sending) and the Express API server (to parse incoming requests).
* **Safe Date & BigInt Coercion**: Configured to parse query params (boolean, number, string) and serialize complex types correctly.

---

## 🚀 Usage

Use these schemas to validate payloads or extract typescript interfaces:

```typescript
import { healthCheckResponse } from "@workspace/api-zod";
import type { HealthStatus } from "@workspace/api-zod";

// Parse and validate an object:
const result = healthCheckResponse.safeParse({ status: "healthy" });
if (result.success) {
  const statusObj: HealthStatus = result.data;
  console.log("Validated successfully:", statusObj);
}
```

---

## 🛠️ Rebuilding the library

To regenerate these validation schemas:

```bash
pnpm --filter @workspace/api-spec run codegen
```
This updates `@workspace/api-zod/src/generated/api.ts` and `@workspace/api-zod/src/generated/types/`.
