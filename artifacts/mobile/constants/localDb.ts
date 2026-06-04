import * as SQLite from "expo-sqlite";

const db = SQLite.openDatabaseSync("nhai_liveness.db");

// Initialize database schema
db.execSync(`
  PRAGMA journal_mode = WAL;

  CREATE TABLE IF NOT EXISTS enrolled_employees (
    employee_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    encrypted_embeddings TEXT NOT NULL,
    is_synced INTEGER DEFAULT 0
  );

  CREATE TABLE IF NOT EXISTS offline_attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    gps TEXT NOT NULL,
    liveness_score REAL NOT NULL,
    embedding_hash TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    is_synced INTEGER DEFAULT 0
  );
`);

export interface EnrolledEmployee {
  employee_id: string;
  name: string;
  encrypted_embeddings: string;
  is_synced: number;
}

export interface OfflineAttendance {
  id: number;
  employee_id: string;
  timestamp: number;
  gps: string;
  liveness_score: number;
  embedding_hash: string;
  record_hash: string;
  is_synced: number;
}

/**
 * Enrolls a new employee locally with their encrypted biometric embeddings.
 */
export async function enrollEmployeeLocal(
  employeeId: string,
  name: string,
  encryptedEmbeddingsHex: string
): Promise<void> {
  await db.runAsync(
    `INSERT OR REPLACE INTO enrolled_employees (employee_id, name, encrypted_embeddings, is_synced)
     VALUES (?, ?, ?, 0);`,
    [employeeId, name, encryptedEmbeddingsHex]
  );
}

/**
 * Retrieves a locally enrolled employee's details (including encrypted embeddings).
 */
export async function getEnrolledEmployeeLocal(
  employeeId: string
): Promise<EnrolledEmployee | null> {
  const row = await db.getFirstAsync<EnrolledEmployee>(
    `SELECT * FROM enrolled_employees WHERE employee_id = ?;`,
    [employeeId]
  );
  return row || null;
}

/**
 * Saves a new attendance record, establishing the blockchain-lite hash link.
 */
export async function saveAttendanceLocal(
  employeeId: string,
  gps: string,
  livenessScore: number,
  embeddingHash: string,
  recordHash: string
): Promise<void> {
  const timestamp = Date.now();
  await db.runAsync(
    `INSERT INTO offline_attendance (employee_id, timestamp, gps, liveness_score, embedding_hash, record_hash, is_synced)
     VALUES (?, ?, ?, ?, ?, ?, 0);`,
    [employeeId, timestamp, gps, livenessScore, embeddingHash, recordHash]
  );
}

/**
 * Gets the record_hash of the latest attendance entry.
 * If no entry exists, returns the genesis hash.
 */
export async function getLatestAttendanceHashLocal(): Promise<string> {
  const row = await db.getFirstAsync<{ record_hash: string }>(
    `SELECT record_hash FROM offline_attendance ORDER BY id DESC LIMIT 1;`
  );
  return row ? row.record_hash : "00000000000000000000000000000000";
}

/**
 * Retrieves all unsynced data to transmit to the server.
 */
export async function getUnsyncedDataLocal(): Promise<{
  enrollments: EnrolledEmployee[];
  attendance: OfflineAttendance[];
}> {
  const enrollments = await db.getAllAsync<EnrolledEmployee>(
    `SELECT * FROM enrolled_employees WHERE is_synced = 0;`
  );
  const attendance = await db.getAllAsync<OfflineAttendance>(
    `SELECT * FROM offline_attendance WHERE is_synced = 0 ORDER BY id ASC;`
  );
  return { enrollments, attendance };
}

/**
 * Purges synced attendance records and enrolled employees (biometric templates) from the device
 * after successful verification/acknowledgment from the server.
 */
export async function purgeSyncedDataLocal(
  employeeIds: string[],
  attendanceIds: number[]
): Promise<void> {
  if (employeeIds.length > 0) {
    const placeholders = employeeIds.map(() => "?").join(",");
    // Securely clear biometric templates from the device but retain employee record
    await db.runAsync(
      `UPDATE enrolled_employees 
       SET encrypted_embeddings = 'SECURELY_CLEARED', is_synced = 1 
       WHERE employee_id IN (${placeholders});`,
      employeeIds
    );
  }

  if (attendanceIds.length > 0) {
    const placeholders = attendanceIds.map(() => "?").join(",");
    // Mark attendance logs as synced instead of deleting them, to preserve device history
    await db.runAsync(
      `UPDATE offline_attendance 
       SET is_synced = 1 
       WHERE id IN (${placeholders});`,
      attendanceIds
    );
  }
}

/**
 * Retrieves all locally enrolled employees.
 */
export async function getAllEmployeesLocal(): Promise<EnrolledEmployee[]> {
  return await db.getAllAsync<EnrolledEmployee>(
    `SELECT * FROM enrolled_employees ORDER BY name ASC;`
  );
}

/**
 * Retrieves all locally stored attendance records.
 */
export async function getAllAttendanceLocal(): Promise<OfflineAttendance[]> {
  return await db.getAllAsync<OfflineAttendance>(
    `SELECT * FROM offline_attendance ORDER BY timestamp DESC;`
  );
}
