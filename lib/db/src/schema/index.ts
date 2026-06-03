import { pgTable, text, timestamp, real, bigint, serial } from "drizzle-orm/pg-core";

export const employeesTable = pgTable("employees", {
  employeeId: text("employee_id").primaryKey(),
  name: text("name").notNull(),
  embeddings: text("embeddings").notNull(), // Serialized JSON array of 512 numbers
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export type Employee = typeof employeesTable.$inferSelect;
export type InsertEmployee = typeof employeesTable.$inferInsert;

export const attendanceTable = pgTable("attendance", {
  id: serial("id").primaryKey(),
  employeeId: text("employee_id").notNull(),
  timestamp: bigint("timestamp", { mode: "number" }).notNull(),
  gps: text("gps").notNull(),
  livenessScore: real("liveness_score").notNull(),
  embeddingHash: text("embedding_hash").notNull(),
  recordHash: text("record_hash").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export type Attendance = typeof attendanceTable.$inferSelect;
export type InsertAttendance = typeof attendanceTable.$inferInsert;