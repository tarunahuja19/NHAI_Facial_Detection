# NHAI Facial Detection & Attendance System

This repository contains the monorepo for the **National Highways Authority of India (NHAI) Facial Detection and Attendance System**. It consists of a cross-platform mobile application that performs on-device face detection, biometric liveness validation, and secure offline log creation, alongside a backend synchronization API server that processes attendance records and aggregates enrolled biometric identities with cryptographic integrity checks.

---

## 🏗️ Architecture Overview

This project is organized as a monorepo managed via `pnpm` workspaces:

```
├── artifacts/
│   ├── api-server/             # Express & TypeScript central API server
│   └── mobile/                 # Expo & React Native mobile client application
├── lib/
│   ├── api-client-react/       # Generated React Query API hooks for the client
│   ├── api-spec/               # OpenAPI specification contract (openapi.yaml)
│   ├── api-zod/                # Generated TypeScript Zod validation schemas
│   └── db/                     # Drizzle ORM database schemas & PG driver config
├── scripts/                    # Utility scripts (database viewer, encrypters)
├── package.json                # Monorepo root configuration
└── pnpm-workspace.yaml         # PNPM workspace configurations
```

---

## 🛠️ Tech Stack & Key Features

### Mobile Application
* **Framework**: React Native with Expo (v54), Expo Router, and TypeScript.
* **On-device ML Engine**: ONNX Runtime (`onnxruntime-react-native`) running optimized face detection (`detector.onnx`) and face recognition (`face_model_quant.onnx`) models.
* **Cryptographic Local Storage**: SQLite database for offline attendance caching, and Expo Secure Store for encrypting face embedding assets with AES key vaults.
* **Multi-Stage Liveness (Anti-Spoofing)**:
  1. **Photometric Stereo**: Flash screen directional light changes to verify face depth/3D structure.
  2. **Moiré Patterns**: ONNX-based texture classification to detect digital screen re-presentation attacks.
  3. **rPPG (Remote Photoplethysmography)**: Measures forehead green color-channel fluctuations to detect real blood volume pulse periodicity.
  4. **IMU Correlation**: Cross-correlates gyroscope motion with face bounding-box motion tracking.

### Backend Server
* **Engine**: Node.js, Express, TypeScript, and Pino Logging.
* **Database & ORM**: PostgreSQL database mapped via Drizzle ORM.
* **Security & Sync Pipeline**:
  * Express endpoints secure by default (removes signature headers, injects strict safety headers like `X-Frame-Options`, `X-Content-Type-Options`).
  * Face model average-merging and re-normalization on conflicts.
  * Blockchain-lite cryptographic chain verification linking consecutive attendance records to prevent data tampering.

---

## 🚀 Getting Started

### Prerequisites
* [Node.js](https://nodejs.org/) (v20+ recommended)
* [pnpm](https://pnpm.io/) (v9+ recommended)
* [PostgreSQL](https://www.postgresql.org/) (running locally or remotely for the api-server)

### Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tarunahuja19/NHAI_Facial_Detection.git
   cd NHAI_Facial_Detection
   ```

2. **Install dependencies:**
   ```bash
   pnpm install
   ```

3. **Configure Environment Variables:**
   Create a `.env` file inside `lib/db/` or where needed for PostgreSQL:
   ```env
   DATABASE_URL=postgresql://user:password@localhost:5432/nhai_db
   ```

4. **Initialize/Push the DB Schema:**
   ```bash
   pnpm --filter @workspace/db run push
   ```

5. **Generate API client code & validations (if needed):**
   ```bash
   pnpm --filter @workspace/api-spec build
   ```

---

## 💻 Workspace Scripts

Run these scripts from the monorepo root:

* **Typecheck all packages**:
  ```bash
  pnpm run typecheck
  ```
* **Build all packages**:
  ```bash
  pnpm run build
  ```
* **Run API server in dev mode**:
  ```bash
  pnpm --filter @workspace/api-server dev
  ```
* **Start mobile expo packager**:
  ```bash
  pnpm --filter @workspace/mobile dev
  ```
