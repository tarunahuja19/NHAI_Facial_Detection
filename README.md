# 🛣️ NHAI Biometric Attendance & Liveness Verification System

Welcome to the **National Highways Authority of India (NHAI) Biometric Attendance & Liveness Verification System** monorepo. This system is designed for high-security, offline-first attendance logging across NHAI toll plaza construction sites and regional offices. 

By combining **on-device Machine Learning (ONNX)** with a **multi-stage cryptographic liveness pipeline** and a **tamper-proof blockchain-lite ledger**, this application guarantees that logs are genuine, non-repudiable, and fully verified before sync.

---

## 🧭 Repository Navigation Guide

To help you navigate the codebase, here is the functional dependency map of the workspace:

```mermaid
graph TD
    api-spec[lib/api-spec: OpenAPI Contract] -->|orval codegen| api-client[lib/api-client-react: React Query Hooks]
    api-spec -->|orval codegen| api-zod[lib/api-zod: Zod Schema Validators]
    
    api-zod -->|Validation Rules| api-server[artifacts/api-server: Express Backend]
    api-zod -->|Schema Matching| mobile[artifacts/mobile: Expo Mobile App]
    api-client -->|HTTP Sync Client| mobile
    
    db[lib/db: Drizzle Postgres Schema] -->|Queries & Updates| api-server
    db -->|Admin Queries| scripts[scripts: Database Inspections]
```

### 📂 Directory Directory Breakdown

#### 1. Core Applications (`/artifacts`)
* **[`artifacts/mobile`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/mobile)**: The user-facing cross-platform Expo application. It accesses the device camera to register employees, executes on-device facial identification, runs a 4-stage hardware/software liveness pipeline, logs logs in a local SQLite file, and schedules network sync.
* **[`artifacts/api-server`](file:///home/jemin/Desktop/Code/nhai-app/artifacts/api-server)**: Node/Express central API backend. Handles delta synchronization, executes employee biometric identity averaging (conflict resolution), and verifies log cryptographic chains before persistence.

#### 2. Shared Libraries & Contracts (`/lib`)
* **[`lib/api-spec`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-spec)**: OpenAPI specification (`openapi.yaml`) mapping network contracts and Orval configurations.
* **[`lib/api-client-react`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-client-react)**: Automated React/React Native networking package exporting custom-configured TanStack queries.
* **[`lib/api-zod`](file:///home/jemin/Desktop/Code/nhai-app/lib/api-zod)**: Shared validation schemas ensuring runtime data compliance across both client and server boundaries.
* **[`lib/db`](file:///home/jemin/Desktop/Code/nhai-app/lib/db)**: Single database controller housing PostgreSQL driver initializers and Drizzle ORM definitions.

#### 3. Administrative Utilities (`/scripts`)
* **[`scripts`](file:///home/jemin/Desktop/Code/nhai-app/scripts)**: Internal developer utility scripts. Includes `view-db.ts` for database table printing and model encryption setups.

---

## 🔍 How the System Works (End-to-End Flow)

Below is the execution flow from the moment an employee checks in to the final server synchronization:

```mermaid
sequenceDiagram
    autonumber
    actor Employee
    participant Mobile as Mobile App (SQLite/ONNX)
    participant Server as Central API Server (Postgres)
    
    Employee->>Mobile: Input Employee ID & Request Attendance
    Note over Mobile: Loads ONNX models (detector, moire, recognizer) into RAM
    
    rect rgb(240, 248, 255)
        Note over Mobile: Multi-Stage Liveness Checks
        Mobile->>Mobile: 1. Photometric Stereo (Flash screen & verify 3D face structure)
        Mobile->>Mobile: 2. ONNX Moiré Check (Validate texture / anti-spoofing)
        Mobile->>Mobile: 3. rPPG Blood Pulse (Assess micro-color variance on forehead)
        Mobile->>Mobile: 4. IMU Correlation (Match gyroscopic motion to face box shift)
    end
    
    Mobile->>Mobile: On-Device Face Recognition (512-dim embedding)
    
    alt Liveness Failed or Cosine Similarity < Threshold
        Mobile-->>Employee: Show Failure Screen
    else Verification Passed
        Note over Mobile: Fetch latest local block hash
        Mobile->>Mobile: Compute record hash: SHA256(prevHash | id | time | embedHash)
        Mobile->>Mobile: Save to Offline SQLite Ledger (is_synced = 0)
        Mobile-->>Employee: Show Success Screen
    end
    
    Note over Mobile: Connection Restored / Click Sync
    Mobile->>Server: POST /api/sync (Payload: Delta Enrollments & Attendance Ledger Chain)
    
    rect rgb(255, 245, 230)
        Note over Server: Server-Side Integrity Check
        Server->>Server: Re-verify full cryptographic chain matching prevHash links
        alt Chain Integrity Broken (Tampering Detected)
            Server-->>Mobile: Reject Sync with 400 Bad Request
        else Chain Valid
            Server->>Server: Merge new biometric templates (Average embeddings on conflict)
            Server->>Server: Insert records to Postgres
            Server-->>Mobile: Respond 200 OK (Acknowledge Synced IDs)
        end
    end
    
    Mobile->>Mobile: Mark attendance as synced. secure-clear local biometric templates!
```

---

## 🛠️ Deep Dive: The Core Systems

### 🛡️ The 4-Stage Liveness Pipeline
Spoofing a standard camera check-in with a photo printout or digital tablet is straightforward. This project implements four overlapping checks to verify presence:
1. **Screen Flash (Photometric Stereo)**: By rapidly flashing the screen white and analyzing brightness variations at the left, right, top, and bottom sectors of the face, the app detects the presence of actual 3D human depth (flat surfaces return uniform reflections).
2. **FFT Moiré Classification**: Captures the face frame, feeds a cropped zone into `moire.onnx`, and detects screen pixel structures.
3. **rPPG Forehead Extraction**: Blood flow causes imperceptible color changes in skin tissue. The camera tracks forehead green-channel intensity changes. A bandpass filter (0.7 Hz to 4.0 Hz) evaluates the periodicity of the signal to identify a human pulse.
4. **IMU motion match**: Correlates the accelerometer and gyroscope data of the physical device with the optical coordinate shift of the face to confirm they move together.

### ⛓️ Blockchain-Lite Ledger
Each local check-in is logged as a "block" containing:
* `employee_id`
* `timestamp`
* `gps` coordinates
* `embedding_hash` (SHA256 of face recognition output)
* `record_hash` (computed as `SHA256(prev_record_hash | employee_id | timestamp | embedding_hash)`)

If any log row is updated, deleted, or inserted out of order, the chain of hashes breaks. The server checks this chain on every synchronization payload.

---

## 🚦 Quick Start for Developers

### Installation & Workspace Build
1. Clone the repository and install all dependencies:
   ```bash
   pnpm install
   ```
2. Build all core libraries and compile TypeScript project maps:
   ```bash
   pnpm run build
   ```

### Running the Backend Server
1. Move to `lib/db` and create a `.env` file containing your PostgreSQL string:
   ```env
   DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/nhai_attendance
   ```
2. Push the schema updates:
   ```bash
   pnpm --filter @workspace/db run push
   ```
3. Boot the Express API Server in development mode:
   ```bash
   pnpm --filter @workspace/api-server dev
   ```

### Running the Mobile Client
1. Boot the Expo development server:
   ```bash
   pnpm --filter @workspace/mobile dev
   ```
2. Scan the QR code with your Expo Go app (Android/iOS) or press `w` to launch the client inside your web browser.
