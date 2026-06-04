# Backend API Server (`@workspace/api-server`)

This folder contains the central synchronization and backend server for the NHAI Facial Detection & Attendance system. It is built using Node.js, Express, TypeScript, and Drizzle ORM, with build bundling managed via `esbuild`.

---

## 🚀 Key Features

* **Biometric Enrollment Merging**: Resolves conflicts when syncing face templates by average-merging 512-dimensional vector embeddings, re-normalizing to unit length, and storing them in the PostgreSQL database.
* **Blockchain-lite Integrity**: Validates the cryptographic hash chain of offline attendance logs. Any tampered or out-of-order records automatically reject synchronization to prevent spoofing of logs.
* **Security Hardening**:
  * Strips framework fingerprint headers (`X-Powered-By`).
  * Injects security headers on every response (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`).
  * Enforces a `15MB` request body limit to mitigate memory exhaustion DoS attacks.
* **Structured Logging**: Employs Pino and `pino-http` logging for high-performance JSON log output, coupled with `pino-pretty` formatting in development.

---

## 📡 API Endpoints

### `GET /api/healthz`
Returns backend health status.
* **Response (200)**:
  ```json
  {
    "status": "healthy"
  }
  ```

### `POST /api/sync`
Synchronizes locally-logged face enrollments and blockchain-lite attendance records.
* **Request Body**:
  ```json
  {
    "enrollments": [
      {
        "employee_id": "EMP123",
        "name": "Jane Doe",
        "embeddings": [ /* Array of 512 numbers */ ]
      }
    ],
    "attendance": [
      {
        "id": 1,
        "employee_id": "EMP123",
        "timestamp": 1717520000000,
        "gps": "28.5355,77.3910",
        "liveness_score": 0.98,
        "embedding_hash": "a4f8c...",
        "record_hash": "c72b8..."
      }
    ]
  }
  ```
* **Response (200)**:
  ```json
  {
    "success": true,
    "syncedEnrollments": ["EMP123"],
    "syncedAttendance": [1]
  }
  ```
* **Response (400)**: If the blockchain hash chain validation fails (tampered logs).
* **Response (500)**: If processing fails.

---

## 🛠️ Commands

Run these inside `artifacts/api-server/` or via monorepo workspace filters:

* **Start Development Mode** (Builds first and runs index):
  ```bash
  pnpm run dev
  ```
* **Compile / Bundle Production App** (via `esbuild`):
  ```bash
  pnpm run build
  ```
* **Start Compiled App**:
  ```bash
  pnpm run start
  ```
* **Typecheck TS Files**:
  ```bash
  pnpm run typecheck
  ```
