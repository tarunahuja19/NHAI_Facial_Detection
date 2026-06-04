# Database & Utility Scripts (`@workspace/scripts`)

This folder contains command-line interface (CLI) inspection and testing scripts helper utilities.

---

## 🛠️ Scripts

Run these scripts from the monorepo root:

### `pnpm --filter @workspace/scripts run view-db`
Connects to the server PostgreSQL database (using the `DATABASE_URL` environment variable) and logs structured lists of:
1. **Registered Employees**: Shows name, employee ID, and a preview of the 512-dimensional face embedding float array.
2. **Attendance Records**: Shows employee, timestamp, GPS position, liveness validation score, biometric hash, and the blockchain-lite record linkage hash.

### `pnpm --filter @workspace/scripts run hello`
Runs a simple smoke test verification script printing greeting logs.

### `pnpm --filter @workspace/scripts run typecheck`
Checks types across script source code.
