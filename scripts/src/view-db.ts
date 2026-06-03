import { db, employeesTable, attendanceTable, type Employee, type Attendance } from "@workspace/db";

async function main() {
  console.log("\n==================================================");
  console.log("   NHAI BIOMETRIC DATABASE INSPECTION UTILITY");
  console.log("==================================================\n");

  try {
    // 1. Fetch enrolled employees
    console.log("--- Enrolled Employees (Biometric Templates) ---");
    const employees = await db.select().from(employeesTable);

    if (employees.length === 0) {
      console.log("No employees currently enrolled on the server.");
    } else {
      const formattedEmployees = employees.map((emp: Employee) => {
        let embeddingsPreview = "[]";
        try {
          const parsed = JSON.parse(emp.embeddings) as number[];
          if (Array.isArray(parsed)) {
            embeddingsPreview = `[${parsed.slice(0, 4).map((v) => v.toFixed(3)).join(", ")} ... total ${parsed.length} floats]`;
          }
        } catch {
          embeddingsPreview = "[Invalid Embeddings JSON]";
        }

        return {
          "Employee ID": emp.employeeId,
          "Full Name": emp.name,
          "Biometric Template Preview": embeddingsPreview,
          "Created At": emp.createdAt ? new Date(emp.createdAt).toLocaleString() : "N/A",
          "Last Updated": emp.updatedAt ? new Date(emp.updatedAt).toLocaleString() : "N/A",
        };
      });
      console.table(formattedEmployees);
    }

    console.log("\n--------------------------------------------------\n");

    // 2. Fetch attendance logs
    console.log("--- Attendance Logs (Blockchain-lite Chained) ---");
    const attendance = await db.select().from(attendanceTable);

    if (attendance.length === 0) {
      console.log("No attendance records currently logged on the server.");
    } else {
      const formattedAttendance = attendance.map((att: Attendance) => ({
        "Record ID": att.id,
        "Employee ID": att.employeeId,
        "Timestamp": new Date(att.timestamp).toLocaleString(),
        "GPS Coordinates": att.gps,
        "Liveness Score": att.livenessScore.toFixed(3),
        "Biometric Hash": att.embeddingHash.substring(0, 16) + "...",
        "Block Chain Hash": att.recordHash.substring(0, 16) + "...",
      }));
      console.table(formattedAttendance);
    }

    console.log("\n==================================================\n");
  } catch (err: any) {
    console.error("Database query failed:", err.message || err);
  } finally {
    process.exit(0);
  }
}

main();
