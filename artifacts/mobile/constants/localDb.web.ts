// Web Fallback: localStorage-backed mock database for web browser support
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

const getEmployees = (): EnrolledEmployee[] => {
  try {
    const data = localStorage.getItem("nhai_employees");
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveEmployees = (list: EnrolledEmployee[]) => {
  localStorage.setItem("nhai_employees", JSON.stringify(list));
};

const getAttendance = (): OfflineAttendance[] => {
  try {
    const data = localStorage.getItem("nhai_attendance");
    return data ? JSON.parse(data) : [];
  } catch {
    return [];
  }
};

const saveAttendanceList = (list: OfflineAttendance[]) => {
  localStorage.setItem("nhai_attendance", JSON.stringify(list));
};

/**
 * Enrolls a new employee locally with their encrypted biometric embeddings.
 */
export async function enrollEmployeeLocal(
  employeeId: string,
  name: string,
  encryptedEmbeddingsHex: string
): Promise<void> {
  const list = getEmployees();
  const existingIdx = list.findIndex(e => e.employee_id === employeeId);
  const record = { employee_id: employeeId, name, encrypted_embeddings: encryptedEmbeddingsHex, is_synced: 0 };
  if (existingIdx >= 0) {
    list[existingIdx] = record;
  } else {
    list.push(record);
  }
  saveEmployees(list);
}

/**
 * Retrieves a locally enrolled employee's details (including encrypted embeddings).
 */
export async function getEnrolledEmployeeLocal(
  employeeId: string
): Promise<EnrolledEmployee | null> {
  return getEmployees().find(e => e.employee_id === employeeId) || null;
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
  const list = getAttendance();
  const id = list.length > 0 ? Math.max(...list.map(a => a.id)) + 1 : 1;
  list.push({ id, employee_id: employeeId, timestamp, gps, liveness_score: livenessScore, embedding_hash: embeddingHash, record_hash: recordHash, is_synced: 0 });
  saveAttendanceList(list);
}

/**
 * Gets the record_hash of the latest attendance entry.
 * If no entry exists, returns the genesis hash.
 */
export async function getLatestAttendanceHashLocal(): Promise<string> {
  const list = getAttendance();
  const row = list.length > 0 ? list[list.length - 1] : null;
  return row ? row.record_hash : "00000000000000000000000000000000";
}

/**
 * Retrieves all unsynced data to transmit to the server.
 */
export async function getUnsyncedDataLocal(): Promise<{
  enrollments: EnrolledEmployee[];
  attendance: OfflineAttendance[];
}> {
  const enrollments = getEmployees().filter(e => e.is_synced === 0);
  const attendance = getAttendance().filter(a => a.is_synced === 0);
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
    const list = getEmployees();
    employeeIds.forEach(id => {
      const emp = list.find(e => e.employee_id === id);
      if (emp) {
        emp.encrypted_embeddings = "SECURELY_CLEARED";
        emp.is_synced = 1;
      }
    });
    saveEmployees(list);
  }

  if (attendanceIds.length > 0) {
    const list = getAttendance();
    attendanceIds.forEach(id => {
      const att = list.find(a => a.id === id);
      if (att) {
        att.is_synced = 1;
      }
    });
    saveAttendanceList(list);
  }
}

/**
 * Retrieves all locally enrolled employees.
 */
export async function getAllEmployeesLocal(): Promise<EnrolledEmployee[]> {
  return getEmployees().sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Retrieves all locally stored attendance records.
 */
export async function getAllAttendanceLocal(): Promise<OfflineAttendance[]> {
  return getAttendance().sort((a, b) => b.timestamp - a.timestamp);
}
