import { Router } from "express";
import crypto from "crypto";
import { db, employeesTable, attendanceTable } from "@workspace/db";
import { eq } from "drizzle-orm";

const router = Router();

function sha256(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

router.post("/sync", async (req, res) => {
  const { enrollments, attendance } = req.body;

  const syncedEnrollments: string[] = [];
  const syncedAttendance: number[] = [];

  try {
    // 1. Process Enrollments with Conflict Resolution (Averaging)
    if (Array.isArray(enrollments)) {
      for (const item of enrollments) {
        const { employee_id, name, embeddings } = item;
        if (!employee_id || !name || !Array.isArray(embeddings) || embeddings.length !== 512) {
          continue;
        }

        // Check if employee already exists in PostgreSQL
        const existing = await db
          .select()
          .from(employeesTable)
          .where(eq(employeesTable.employeeId, employee_id))
          .limit(1);

        if (existing.length > 0) {
          // Conflict resolution: average embeddings (robust check on JSON parse)
          let existingEmbeddings: number[] = [];
          try {
            existingEmbeddings = JSON.parse(existing[0].embeddings) as number[];
          } catch (e) {
            existingEmbeddings = [];
          }

          if (Array.isArray(existingEmbeddings) && existingEmbeddings.length === 512) {
            const merged = new Array(512).fill(0);
            for (let i = 0; i < 512; i++) {
              merged[i] = (existingEmbeddings[i] + embeddings[i]) / 2;
            }
            // Re-normalize merged vector to unit length
            const norm = Math.sqrt(merged.reduce((sum, v) => sum + v * v, 0));
            const normalized = merged.map((v) => v / (norm || 1));

            await db
              .update(employeesTable)
              .set({
                name,
                embeddings: JSON.stringify(normalized),
                updatedAt: new Date(),
              })
              .where(eq(employeesTable.employeeId, employee_id));
          } else {
            // Fallback: If existing template is malformed, overwrite with new valid enrollment
            const norm = Math.sqrt(embeddings.reduce((sum, v) => sum + v * v, 0));
            const normalized = embeddings.map((v) => v / (norm || 1));

            await db
              .update(employeesTable)
              .set({
                name,
                embeddings: JSON.stringify(normalized),
                updatedAt: new Date(),
              })
              .where(eq(employeesTable.employeeId, employee_id));
          }
        } else {
          // New employee enrollment
          const norm = Math.sqrt(embeddings.reduce((sum, v) => sum + v * v, 0));
          const normalized = embeddings.map((v) => v / (norm || 1));

          await db.insert(employeesTable).values({
            employeeId: employee_id,
            name,
            embeddings: JSON.stringify(normalized),
          });
        }
        syncedEnrollments.push(employee_id);
      }
    }

    // 2. Process Attendance with Blockchain-lite Integrity Verification
    if (Array.isArray(attendance) && attendance.length > 0) {
      // Sort chronologically by timestamp
      const sortedAttendance = [...attendance].sort((a, b) => a.timestamp - b.timestamp);

      // Verify the integrity of the chain
      let isChainValid = true;
      for (let i = 0; i < sortedAttendance.length; i++) {
        const record = sortedAttendance[i];
        const prevHash = i === 0 ? "00000000000000000000000000000000" : sortedAttendance[i - 1].record_hash;

        const chainInput = `${prevHash}|${record.employee_id}|${record.timestamp}|${record.embedding_hash}`;
        const calculatedHash = sha256(chainInput);

        if (calculatedHash !== record.record_hash) {
          isChainValid = false;
          break;
        }
      }

      if (!isChainValid) {
        res.status(400).json({ error: "Tampered or invalid attendance hash chain detected." });
        return;
      }

      // Save valid attendance records to PostgreSQL
      for (const record of sortedAttendance) {
        await db.insert(attendanceTable).values({
          employeeId: record.employee_id,
          timestamp: record.timestamp,
          gps: record.gps,
          livenessScore: record.liveness_score,
          embeddingHash: record.embedding_hash,
          recordHash: record.record_hash,
        });
        syncedAttendance.push(record.id);
      }
    }

    res.json({
      success: true,
      syncedEnrollments,
      syncedAttendance,
    });
  } catch (err: any) {
    console.error("Sync API Error:", err);
    res.status(500).json({ error: err.message || "Failed to process synchronization delta." });
  }
});

export default router;
