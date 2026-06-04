# Shared Database Access Layer (`@workspace/db`)

This package houses the database configurations, schemas, and ORM clients mapping the PostgreSQL database tables using [Drizzle ORM](https://orm.drizzle.team/).

---

## 🏗️ Schema Definition

### 1. `employees` table (`employeesTable`)
Contains registered worker profiles and their averaged biometric templates:
* `employee_id` (TEXT, PK): Unique identification of the employee.
* `name` (TEXT): Name of the employee.
* `embeddings` (TEXT): Serialized JSON array of 512 numbers representing the face model template.
* `created_at` / `updated_at` (TIMESTAMP): Timestamps.

### 2. `attendance` table (`attendanceTable`)
Contains logged check-ins with blockchain-lite cryptographic chains:
* `id` (SERIAL, PK): Unique record identifier.
* `employee_id` (TEXT): ID of the worker.
* `timestamp` (BIGINT): Unix epoch timestamp of checking.
* `gps` (TEXT): Coordinates string (Latitude,Longitude).
* `liveness_score` (REAL): Liveness verification pass probability.
* `embedding_hash` (TEXT): Hash of the captured face embedding vector.
* `record_hash` (TEXT): Cryptographic hash linking this check-in to the previous record's hash, establishing an immutable chain.

---

## 🚀 Usage

```typescript
import { db, employeesTable } from "@workspace/db";
import { eq } from "drizzle-orm";

// Query employees:
const employee = await db
  .select()
  .from(employeesTable)
  .where(eq(employeesTable.employeeId, "EMP123"));
```

---

## 🛠️ Database Setup & Migrations

Configure your environment:
* Ensure `DATABASE_URL` is set in your environment variables. Example:
  ```bash
  export DATABASE_URL="postgresql://postgres:password@localhost:5432/nhai_db"
  ```

Run these scripts within `lib/db/` or from the monorepo root:

* **Push Schema to DB**:
  ```bash
  pnpm run push
  ```
* **Force Schema Sync** (if resolving breaking conflicts in dev):
  ```bash
  pnpm run push-force
  ```
