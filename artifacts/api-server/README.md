# 🖥️ NHAI Backend API Server (`@workspace/api-server`)

This package houses the central TypeScript API server responsible for receiving attendance logs, verifying biometric authenticity, handling client registration conflict resolutions, and maintaining the master PostgreSQL attendance ledger.

---

## 📂 Navigation & File Structure

```
├── src/
│   ├── index.ts               # Entrypoint (Boots Express server and binds to port)
│   ├── app.ts                 # Express application initialization & middleware stacks
│   ├── lib/
│   │   └── logger.ts          # Pino logger config for high-speed JSON output
│   ├── middlewares/           # Core request interceptors
│   └── routes/
│       ├── index.ts           # Central router registering sub-routes
│       ├── health.ts          # Endpoint checking DB/server status (/healthz)
│       └── sync.ts            # Synchronization handler (/sync)
├── build.mjs                  # ESBuild script compilation schema
├── package.json               # Package commands & dependency listing
└── tsconfig.json              # TypeScript compilation rules
```

---

## ⚙️ Core Technical Capabilities

### 🛡️ 1. Hardened API Security Middleware
To protect governmental employee logs, the server enforces structural hardening in [`src/app.ts`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/api-server/src/app.ts):
* **Express Cloaking**: Strips the default `X-Powered-By` header to prevent scanning scripts from identifying target server technologies.
* **Manually Injected Security Headers**:
  * `X-Content-Type-Options: nosniff` (Prevents MIME sniffing attacks).
  * `X-Frame-Options: DENY` (Mitigates clickjacking overlays).
  * `X-XSS-Protection: 1; mode=block` (Blocks cross-site scripting page renders).
  * `Referrer-Policy: no-referrer` (Prevents credential leakages in referrer headers).
* **Payload Size Constraints**: Enforces a strict `15MB` request body limitation. Since biometric embedding payloads contain raw arrays of floating points, this prevents heap memory depletion/DoS vectors.

### 🧬 2. Average-Merge Biometric Conflict Resolution
In [`src/routes/sync.ts`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/api-server/src/routes/sync.ts), when an employee re-registers or registers across different plaza devices:
1. The server checks the existing database template for that employee ID.
2. If a template exists, it retrieves the previous 512-dimensional vector.
3. It performs a **Vector Averaging merge**: `merged[i] = (existing[i] + new[i]) / 2`.
4. It calculates the Euclidean Norm of the merged vector and **re-normalizes it to unit length** (unit length is required for cosine similarity operations on the mobile app).
5. Updates the template in PostgreSQL. This smooths out camera sensor variances from different capture devices.

### ⛓️ 3. Blockchain-Lite Verification
To prevent offline database injection:
1. The client submits a chronological array of attendance records.
2. The server iterates over each record and computes:
   `calculatedHash = SHA256(prevHash | employee_id | timestamp | embedding_hash)`
3. It matches this value against the client's provided `record_hash`.
4. If there is a mismatch at any index, the server aborts the transaction, returns a `400 Bad Request`, and writes warning logs to Pino.

---

## 📡 API Specs

### `GET /api/healthz`
Health monitor checking backend liveliness.
* **Response (200)**:
  ```json
  {
    "status": "healthy"
  }
  ```

### `POST /api/sync`
Synchronizes newly registered profiles and attendance events.
* **Payload Structure**:
  ```json
  {
    "enrollments": [
      {
        "employee_id": "EMP007",
        "name": "Arjun Singh",
        "embeddings": [0.012, -0.045, ... 512 floats]
      }
    ],
    "attendance": [
      {
        "employee_id": "EMP007",
        "timestamp": 1780577405000,
        "gps": "28.5355,77.3910",
        "liveness_score": 0.992,
        "embedding_hash": "e3b0c442...",
        "record_hash": "8f482a..."
      }
    ]
  }
  ```
* **Success (200)**:
  ```json
  {
    "success": true,
    "syncedEnrollments": ["EMP007"],
    "syncedAttendance": [1]
  }
  ```

---

## 🛠️ Developer Scripts

Run these scripts locally to manage the server:

* **Local dev start** (Hot builds and starts built file with source maps enabled):
  ```bash
  pnpm run dev
  ```
* **Compile production package** (Uses ESBuild and Pinobuild bundle extensions):
  ```bash
  pnpm run build
  ```
* **Run static built file**:
  ```bash
  pnpm run start
  ```
