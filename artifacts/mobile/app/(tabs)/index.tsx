import * as Haptics from "expo-haptics";
import { Image } from "expo-image";
import { router, useFocusEffect } from "expo-router";
import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  ActivityIndicator,
  Alert,
  Animated,
  AppState,
  AppStateStatus,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
  Modal,
  KeyboardAvoidingView,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import { Feather } from "@expo/vector-icons";

import { useColors } from "@/hooks/useColors";
import {
  getUnsyncedDataLocal,
  purgeSyncedDataLocal,
  getAllEmployeesLocal,
  getAllAttendanceLocal,
  EnrolledEmployee,
  OfflineAttendance,
} from "@/constants/localDb";
import { decryptEmbeddings } from "@/constants/vault";

// ─── Types ────────────────────────────────────────────────────────────────────

interface SyncResult {
  timestamp: number;
  enrollmentsSynced: string[];   // employee IDs
  attendanceSynced: number;      // count
  error?: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const formatTimestamp = (ts: number) => {
  const date = new Date(ts);
  const months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  const m = months[date.getMonth()];
  const d = date.getDate();
  let hr = date.getHours();
  const min = String(date.getMinutes()).padStart(2, "0");
  const ampm = hr >= 12 ? "PM" : "AM";
  hr = hr % 12;
  hr = hr ? hr : 12;
  return `${d} ${m}, ${hr}:${min} ${ampm}`;
};

/** Lightweight connectivity check — tries to reach the server health endpoint */
async function checkServerReachable(serverUrl: string): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
    const cleanUrl = serverUrl.replace(/\/+$/, "");
    const res = await fetch(`${cleanUrl}/api/healthz`, {
      method: "GET",
      signal: controller.signal,
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function HomeScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();

  const logoAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;

  // Tab State
  const [activeTab, setActiveTab] = useState<"attendance" | "workers" | "sync">("attendance");

  // Database Records
  const [employees, setEmployees] = useState<EnrolledEmployee[]>([]);
  const [attendanceLogs, setAttendanceLogs] = useState<OfflineAttendance[]>([]);
  const [searchQuery, setSearchQuery] = useState("");

  // Sync State
  const [serverUrl, setServerUrl] = useState("http://10.0.2.2:3000");
  const [isSyncing, setIsSyncing] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [unsyncedEnrollments, setUnsyncedEnrollments] = useState(0);
  const [unsyncedAttendance, setUnsyncedAttendance] = useState(0);
  const [syncHistory, setSyncHistory] = useState<SyncResult[]>([]);
  const [lastSyncTime, setLastSyncTime] = useState<number | null>(null);

  // Modals
  const [showVerifyModal, setShowVerifyModal] = useState(false);
  const [showEnrollModal, setShowEnrollModal] = useState(false);
  const [showServerModal, setShowServerModal] = useState(false);
  const [employeeId, setEmployeeId] = useState("");
  const [employeeName, setEmployeeName] = useState("");
  const [serverUrlDraft, setServerUrlDraft] = useState(serverUrl);

  // Refs to avoid stale closures in intervals
  const serverUrlRef = useRef(serverUrl);
  const isSyncingRef = useRef(isSyncing);
  serverUrlRef.current = serverUrl;
  isSyncingRef.current = isSyncing;

  // ── Animations ──────────────────────────────────────────────────────────────
  useEffect(() => {
    Animated.parallel([
      Animated.timing(logoAnim, { toValue: 1, duration: 700, useNativeDriver: false }),
      Animated.timing(fadeAnim,  { toValue: 1, duration: 700, useNativeDriver: false }),
    ]).start();
  }, []);

  // ── Data Refresh ────────────────────────────────────────────────────────────
  const refreshData = useCallback(async () => {
    try {
      const unsynced = await getUnsyncedDataLocal();
      setUnsyncedEnrollments(unsynced.enrollments.length);
      setUnsyncedAttendance(unsynced.attendance.length);
      const emps = await getAllEmployeesLocal();
      const logs = await getAllAttendanceLocal();
      setEmployees(emps);
      setAttendanceLogs(logs);
    } catch (err) {
      console.warn("Failed to load local DB data:", err);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      refreshData();
    }, [refreshData])
  );

  // ── Core Sync Logic ─────────────────────────────────────────────────────────
  const performSync = useCallback(async (silent = false): Promise<boolean> => {
    if (isSyncingRef.current) return false;
    setIsSyncing(true);

    try {
      const localData = await getUnsyncedDataLocal();

      if (localData.enrollments.length === 0 && localData.attendance.length === 0) {
        if (!silent) Alert.alert("Up to Date", "All records are already synced with the server.");
        setIsSyncing(false);
        return true;
      }

      // Decrypt enrollment embeddings for server
      const enrollmentsWithEmbeddings = await Promise.all(
        localData.enrollments.map(async (e) => {
          const embeddings = await decryptEmbeddings(e.encrypted_embeddings);
          return { employee_id: e.employee_id, name: e.name, embeddings };
        })
      );

      const payload = {
        enrollments: enrollmentsWithEmbeddings,
        attendance: localData.attendance,
      };

      const cleanUrl = serverUrlRef.current.replace(/\/+$/, "");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 15000);
      let res: Response;
      try {
        res = await fetch(`${cleanUrl}/api/sync`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timer);
      }

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Server error ${res.status}: ${body}`);
      }

      const syncResult = await res.json();

      const syncedEmployeeIds: string[]  = syncResult.syncedEnrollments || [];
      const syncedAttendanceIds: number[] = syncResult.syncedAttendance  || [];

      await purgeSyncedDataLocal(syncedEmployeeIds, syncedAttendanceIds);
      await refreshData();

      const result: SyncResult = {
        timestamp: Date.now(),
        enrollmentsSynced: syncedEmployeeIds,
        attendanceSynced: syncedAttendanceIds.length,
      };

      setSyncHistory(prev => [result, ...prev].slice(0, 20));
      setLastSyncTime(Date.now());

      if (!silent) {
        Alert.alert(
          "✅ Sync Complete",
          `Uploaded to database:\n• ${syncedEmployeeIds.length} worker(s): ${syncedEmployeeIds.join(", ") || "none"}\n• ${syncedAttendanceIds.length} attendance record(s)`
        );
      }

      return true;
    } catch (err: any) {
      const result: SyncResult = {
        timestamp: Date.now(),
        enrollmentsSynced: [],
        attendanceSynced: 0,
        error: err.message || "Unknown error",
      };
      setSyncHistory(prev => [result, ...prev].slice(0, 20));

      if (!silent) Alert.alert("Sync Failed", err.message || "Could not reach the server.");
      return false;
    } finally {
      setIsSyncing(false);
    }
  }, [refreshData]);

  // ── Auto-sync: connectivity polling ─────────────────────────────────────────
  // Every 30s check if server is reachable; if yes & we have unsynced data → sync
  useEffect(() => {
    let pollTimer: ReturnType<typeof setInterval> | null = null;

    const checkAndSync = async () => {
      const reachable = await checkServerReachable(serverUrlRef.current);
      setIsOnline(reachable);
      if (reachable && (unsyncedEnrollments > 0 || unsyncedAttendance > 0)) {
        await performSync(true);
      }
    };

    // Check on mount + when app comes to foreground
    checkAndSync();

    const sub = AppState.addEventListener("change", (state: AppStateStatus) => {
      if (state === "active") checkAndSync();
    });

    // Poll every 30 seconds
    pollTimer = setInterval(checkAndSync, 30_000);

    return () => {
      if (pollTimer) clearInterval(pollTimer);
      sub.remove();
    };
  }, [unsyncedEnrollments, unsyncedAttendance, performSync]);

  // ── Navigation ───────────────────────────────────────────────────────────────
  const handleStartLiveness = (mode: "verify" | "enroll") => {
    if (!employeeId.trim()) {
      Alert.alert("Required", "Please enter a valid Employee ID.");
      return;
    }
    if (mode === "enroll" && !employeeName.trim()) {
      Alert.alert("Required", "Please enter the Employee Name.");
      return;
    }

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setShowVerifyModal(false);
    setShowEnrollModal(false);

    router.push({
      pathname: "/liveness",
      params: {
        mode,
        employeeId: employeeId.trim(),
        employeeName: mode === "enroll" ? employeeName.trim() : "",
      },
    });

    setEmployeeId("");
    setEmployeeName("");
  };

  // ── Derived ──────────────────────────────────────────────────────────────────
  const filteredEmployees = employees.filter(
    (e) =>
      e.employee_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      e.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredAttendanceLogs = attendanceLogs.filter((log) =>
    log.employee_id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const topInset    = Platform.OS === "web" ? 67 : insets.top;
  const bottomInset = Platform.OS === "web" ? 34 : insets.bottom;
  const totalUnsynced = unsyncedEnrollments + unsyncedAttendance;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>

      {/* ── Header ── */}
      <View style={[styles.headerBar, { paddingTop: topInset + 8, backgroundColor: colors.card, borderBottomColor: colors.border }]}>
        <View style={styles.headerLeft}>
          <Image
            source={require("@/assets/images/nhai_logo.jpg")}
            style={styles.logoMini}
            contentFit="contain"
          />
          <View>
            <Text style={[styles.headerTitle, { color: colors.foreground }]}>NHAI</Text>
            <Text style={[styles.headerSubtitle, { color: colors.mutedForeground }]}>Biometric Portal</Text>
          </View>
        </View>

        {/* Online indicator + sync button */}
        <View style={styles.syncContainerHeader}>
          <View style={[styles.onlineDot, { backgroundColor: isOnline ? "#22c55e" : "#94a3b8" }]} />

          {totalUnsynced > 0 && (
            <View style={[styles.badgeUnsynced, { backgroundColor: "#f97316" }]}>
              <Text style={styles.badgeText}>{totalUnsynced}</Text>
            </View>
          )}

          <Pressable
            onPress={() => performSync(false)}
            style={({ pressed }) => [
              styles.syncIconBtn,
              { backgroundColor: colors.primary + "15", opacity: pressed || isSyncing ? 0.6 : 1 },
            ]}
            disabled={isSyncing}
          >
            {isSyncing ? (
              <ActivityIndicator size="small" color={colors.primary} />
            ) : (
              <Feather name="refresh-cw" size={18} color={colors.primary} />
            )}
          </Pressable>

          <Pressable
            onPress={() => {
              setServerUrlDraft(serverUrl);
              setShowServerModal(true);
            }}
            style={[styles.syncIconBtn, { backgroundColor: colors.mutedForeground + "15" }]}
          >
            <Feather name="settings" size={16} color={colors.mutedForeground} />
          </Pressable>
        </View>
      </View>

      {/* ── Status Bar ── */}
      {(isSyncing || lastSyncTime) && (
        <View style={[styles.statusBar, { backgroundColor: isSyncing ? "#1e40af15" : "#16a34a15", borderBottomColor: colors.border }]}>
          {isSyncing ? (
            <>
              <ActivityIndicator size="small" color="#1e40af" style={{ marginRight: 8 }} />
              <Text style={[styles.statusBarText, { color: "#1e40af" }]}>Syncing with server…</Text>
            </>
          ) : lastSyncTime ? (
            <>
              <Feather name="check-circle" size={13} color="#16a34a" style={{ marginRight: 6 }} />
              <Text style={[styles.statusBarText, { color: "#16a34a" }]}>
                Last synced {formatTimestamp(lastSyncTime)}
              </Text>
            </>
          ) : null}
        </View>
      )}

      {/* ── Tab Bar ── */}
      <View style={[styles.tabContainer, { borderBottomColor: colors.border }]}>
        {(["attendance", "workers", "sync"] as const).map((tab) => {
          const icons = { attendance: "check-square", workers: "users", sync: "cloud" } as const;
          const labels = { attendance: "Attendance", workers: "Workers", sync: `DB Sync${syncHistory.length > 0 ? ` (${syncHistory.length})` : ""}` };
          const isActive = activeTab === tab;
          return (
            <Pressable
              key={tab}
              onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setActiveTab(tab); }}
              style={[styles.tabItem, isActive && { borderBottomColor: colors.primary }]}
            >
              <Feather name={icons[tab]} size={15} color={isActive ? colors.primary : colors.mutedForeground} />
              <Text style={[styles.tabLabel, { color: isActive ? colors.primary : colors.mutedForeground }]}>
                {labels[tab]}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* ── Search ── */}
      {activeTab !== "sync" && (
        <View style={styles.searchContainer}>
          <View style={[styles.searchBox, { backgroundColor: colors.card, borderColor: colors.border }]}>
            <Feather name="search" size={16} color={colors.mutedForeground} style={{ marginRight: 8 }} />
            <TextInput
              placeholder={activeTab === "attendance" ? "Search by Employee ID…" : "Search by ID or Name…"}
              placeholderTextColor={colors.mutedForeground}
              value={searchQuery}
              onChangeText={setSearchQuery}
              style={[styles.searchInput, { color: colors.foreground }]}
            />
            {searchQuery.length > 0 && (
              <Pressable onPress={() => setSearchQuery("")}>
                <Feather name="x" size={16} color={colors.mutedForeground} />
              </Pressable>
            )}
          </View>
        </View>
      )}

      {/* ── Content ── */}
      <ScrollView
        contentContainerStyle={[styles.listScroll, { paddingBottom: bottomInset + 100 }]}
        showsVerticalScrollIndicator={false}
      >

        {/* ATTENDANCE TAB */}
        {activeTab === "attendance" && (
          filteredAttendanceLogs.length === 0 ? (
            <View style={styles.placeholderContainer}>
              <Feather name="clipboard" size={48} color={colors.mutedForeground + "40"} />
              <Text style={[styles.placeholderTitle, { color: colors.foreground }]}>No Attendance Logs</Text>
              <Text style={[styles.placeholderText, { color: colors.mutedForeground }]}>
                {searchQuery.length > 0 ? "No matching logs found." : "Tap Authenticate Employee below to record attendance."}
              </Text>
            </View>
          ) : (
            filteredAttendanceLogs.map((log) => (
              <View key={log.id} style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderLeft}>
                    <View style={[styles.iconCircle, { backgroundColor: colors.primary + "15" }]}>
                      <Feather name="user-check" size={16} color={colors.primary} />
                    </View>
                    <Text style={[styles.cardTitle, { color: colors.foreground }]}>{log.employee_id}</Text>
                  </View>
                  <View style={[styles.statusPill, log.is_synced === 1 ? styles.pillGreen : styles.pillOrange]}>
                    <Text style={[styles.statusPillText, { color: log.is_synced === 1 ? "#166534" : "#9a3412" }]}>
                      {log.is_synced === 1 ? "✓ Synced" : "⏳ Offline"}
                    </Text>
                  </View>
                </View>

                <View style={styles.cardDivider} />

                <View style={styles.cardRow}>
                  <View style={styles.cardCol}>
                    <Text style={[styles.cardLabel, { color: colors.mutedForeground }]}>Timestamp</Text>
                    <Text style={[styles.cardValue, { color: colors.foreground }]}>{formatTimestamp(log.timestamp)}</Text>
                  </View>
                  <View style={styles.cardCol}>
                    <Text style={[styles.cardLabel, { color: colors.mutedForeground }]}>Liveness Score</Text>
                    <Text style={[styles.cardValue, { color: colors.primary, fontWeight: "bold" }]}>
                      {(log.liveness_score * 100).toFixed(0)}%
                    </Text>
                  </View>
                </View>

                <View style={[styles.cardRow, { marginTop: 8 }]}>
                  <View style={styles.cardCol}>
                    <Text style={[styles.cardLabel, { color: colors.mutedForeground }]}>GPS</Text>
                    <Text style={[styles.cardValue, { color: colors.foreground }]}>{log.gps}</Text>
                  </View>
                </View>
              </View>
            ))
          )
        )}

        {/* WORKERS TAB */}
        {activeTab === "workers" && (
          filteredEmployees.length === 0 ? (
            <View style={styles.placeholderContainer}>
              <Feather name="users" size={48} color={colors.mutedForeground + "40"} />
              <Text style={[styles.placeholderTitle, { color: colors.foreground }]}>No Enrolled Workers</Text>
              <Text style={[styles.placeholderText, { color: colors.mutedForeground }]}>
                {searchQuery.length > 0 ? "No matching employees found." : "Tap Enroll New Worker below to register employee biometrics."}
              </Text>
            </View>
          ) : (
            filteredEmployees.map((emp) => (
              <View key={emp.employee_id} style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                <View style={styles.cardHeader}>
                  <View style={styles.cardHeaderLeft}>
                    <View style={[styles.iconCircle, { backgroundColor: colors.primary + "15" }]}>
                      <Feather name="user" size={16} color={colors.primary} />
                    </View>
                    <View>
                      <Text style={[styles.cardTitle, { color: colors.foreground }]}>{emp.name}</Text>
                      <Text style={[styles.cardSubText, { color: colors.mutedForeground }]}>ID: {emp.employee_id}</Text>
                    </View>
                  </View>
                  <View style={[styles.statusPill, emp.is_synced === 1 ? styles.pillGreen : styles.pillOrange]}>
                    <Text style={[styles.statusPillText, { color: emp.is_synced === 1 ? "#166534" : "#9a3412" }]}>
                      {emp.is_synced === 1 ? "✓ Synced" : "Enrolled"}
                    </Text>
                  </View>
                </View>

                {emp.is_synced === 1 && (
                  <View style={[styles.cardFooter, { backgroundColor: "#16a34a10" }]}>
                    <Feather name="shield" size={12} color="#166534" style={{ marginRight: 6 }} />
                    <Text style={[styles.footerText, { color: "#166534" }]}>
                      Biometrics uploaded to database. Template cleared from device.
                    </Text>
                  </View>
                )}
              </View>
            ))
          )
        )}

        {/* SYNC HISTORY TAB */}
        {activeTab === "sync" && (
          <>
            {/* Current pending summary */}
            <View style={[styles.syncSummaryCard, { backgroundColor: colors.card, borderColor: colors.border }]}>
              <Text style={[styles.syncSummaryTitle, { color: colors.foreground }]}>Pending Upload</Text>
              <View style={styles.syncSummaryRow}>
                <View style={styles.syncSummaryItem}>
                  <Text style={[styles.syncSummaryNum, { color: unsyncedEnrollments > 0 ? "#f97316" : "#22c55e" }]}>
                    {unsyncedEnrollments}
                  </Text>
                  <Text style={[styles.syncSummaryLabel, { color: colors.mutedForeground }]}>Workers</Text>
                </View>
                <View style={[styles.syncSummaryDivider, { backgroundColor: colors.border }]} />
                <View style={styles.syncSummaryItem}>
                  <Text style={[styles.syncSummaryNum, { color: unsyncedAttendance > 0 ? "#f97316" : "#22c55e" }]}>
                    {unsyncedAttendance}
                  </Text>
                  <Text style={[styles.syncSummaryLabel, { color: colors.mutedForeground }]}>Attendance</Text>
                </View>
                <View style={[styles.syncSummaryDivider, { backgroundColor: colors.border }]} />
                <View style={styles.syncSummaryItem}>
                  <View style={[styles.onlineDot, { backgroundColor: isOnline ? "#22c55e" : "#94a3b8", width: 10, height: 10, marginBottom: 4 }]} />
                  <Text style={[styles.syncSummaryLabel, { color: colors.mutedForeground }]}>
                    {isOnline ? "Online" : "Offline"}
                  </Text>
                </View>
              </View>

              <Pressable
                onPress={() => performSync(false)}
                disabled={isSyncing}
                style={({ pressed }) => [
                  styles.syncNowBtn,
                  { backgroundColor: colors.primary, opacity: pressed || isSyncing ? 0.7 : 1 },
                ]}
              >
                {isSyncing ? (
                  <ActivityIndicator size="small" color="#fff" style={{ marginRight: 8 }} />
                ) : (
                  <Feather name="upload-cloud" size={16} color="#fff" style={{ marginRight: 8 }} />
                )}
                <Text style={styles.syncNowBtnText}>
                  {isSyncing ? "Uploading…" : "Sync Now"}
                </Text>
              </Pressable>
            </View>

            {/* Sync History */}
            <Text style={[styles.sectionHeader, { color: colors.mutedForeground }]}>Upload History</Text>

            {syncHistory.length === 0 ? (
              <View style={styles.placeholderContainer}>
                <Feather name="cloud" size={48} color={colors.mutedForeground + "40"} />
                <Text style={[styles.placeholderTitle, { color: colors.foreground }]}>No Sync History Yet</Text>
                <Text style={[styles.placeholderText, { color: colors.mutedForeground }]}>
                  Tap Sync Now above or connect to the internet to auto-sync.
                </Text>
              </View>
            ) : (
              syncHistory.map((item, idx) => (
                <View key={idx} style={[styles.card, { backgroundColor: colors.card, borderColor: colors.border }]}>
                  <View style={styles.cardHeader}>
                    <View style={styles.cardHeaderLeft}>
                      <View style={[styles.iconCircle, { backgroundColor: item.error ? "#fef2f2" : "#f0fdf4" }]}>
                        <Feather
                          name={item.error ? "alert-circle" : "check-circle"}
                          size={16}
                          color={item.error ? "#dc2626" : "#16a34a"}
                        />
                      </View>
                      <View>
                        <Text style={[styles.cardTitle, { color: colors.foreground }]}>
                          {item.error ? "Sync Failed" : "Upload Successful"}
                        </Text>
                        <Text style={[styles.cardSubText, { color: colors.mutedForeground }]}>
                          {formatTimestamp(item.timestamp)}
                        </Text>
                      </View>
                    </View>
                  </View>

                  <View style={styles.cardDivider} />

                  {item.error ? (
                    <Text style={[styles.cardValue, { color: "#dc2626", fontSize: 12 }]}>{item.error}</Text>
                  ) : (
                    <>
                      <View style={styles.cardRow}>
                        <View style={styles.cardCol}>
                          <Text style={[styles.cardLabel, { color: colors.mutedForeground }]}>Workers Uploaded</Text>
                          <Text style={[styles.cardValue, { color: colors.foreground }]}>
                            {item.enrollmentsSynced.length > 0 ? item.enrollmentsSynced.join(", ") : "None"}
                          </Text>
                        </View>
                      </View>
                      <View style={[styles.cardRow, { marginTop: 8 }]}>
                        <View style={styles.cardCol}>
                          <Text style={[styles.cardLabel, { color: colors.mutedForeground }]}>Attendance Records</Text>
                          <Text style={[styles.cardValue, { color: colors.foreground }]}>
                            {item.attendanceSynced} record{item.attendanceSynced !== 1 ? "s" : ""} uploaded
                          </Text>
                        </View>
                      </View>
                      <View style={[styles.cardFooter, { backgroundColor: "#f0fdf4", marginTop: 10 }]}>
                        <Feather name="database" size={12} color="#16a34a" style={{ marginRight: 6 }} />
                        <Text style={[styles.footerText, { color: "#16a34a" }]}>
                          Data saved to PostgreSQL database on server.
                        </Text>
                      </View>
                    </>
                  )}
                </View>
              ))
            )}
          </>
        )}
      </ScrollView>

      {/* ── FAB ── */}
      {activeTab !== "sync" && (
        <View style={[styles.fabContainer, { bottom: bottomInset + 16 }]}>
          {activeTab === "attendance" ? (
            <Pressable
              onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setEmployeeId(""); setShowVerifyModal(true); }}
              style={[styles.primaryFab, { backgroundColor: colors.primary }]}
            >
              <Feather name="check-square" size={20} color="#fff" style={{ marginRight: 8 }} />
              <Text style={styles.primaryFabText}>Authenticate Employee</Text>
            </Pressable>
          ) : (
            <Pressable
              onPress={() => { Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light); setEmployeeId(""); setEmployeeName(""); setShowEnrollModal(true); }}
              style={[styles.primaryFab, { backgroundColor: colors.primary }]}
            >
              <Feather name="user-plus" size={20} color="#fff" style={{ marginRight: 8 }} />
              <Text style={styles.primaryFabText}>Enroll New Worker</Text>
            </Pressable>
          )}
        </View>
      )}

      {/* ── Verify Modal ── */}
      <Modal visible={showVerifyModal} transparent animationType="fade" onRequestClose={() => setShowVerifyModal(false)}>
        <View style={styles.modalBackdrop}>
          <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.modalWrapper}>
            <View style={[styles.modalCard, { backgroundColor: colors.card }]}>
              <View style={styles.modalHeader}>
                <Text style={[styles.modalTitle, { color: colors.foreground }]}>Record Attendance</Text>
                <Pressable onPress={() => setShowVerifyModal(false)} style={styles.closeBtn}>
                  <Feather name="x" size={20} color={colors.mutedForeground} />
                </Pressable>
              </View>
              <Text style={[styles.modalDesc, { color: colors.mutedForeground }]}>
                Enter the employee ID to start facial liveness verification.
              </Text>
              <Text style={[styles.fieldLabel, { color: colors.foreground }]}>Employee ID</Text>
              <TextInput
                value={employeeId}
                onChangeText={setEmployeeId}
                placeholder="e.g. NHAI-4029"
                placeholderTextColor={colors.mutedForeground}
                style={[styles.textInput, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
                autoCapitalize="characters"
                autoCorrect={false}
                autoFocus
              />
              <View style={styles.modalButtons}>
                <Pressable onPress={() => setShowVerifyModal(false)} style={[styles.modalBtn, styles.cancelBtn, { borderColor: colors.border }]}>
                  <Text style={[styles.cancelBtnText, { color: colors.foreground }]}>Cancel</Text>
                </Pressable>
                <Pressable onPress={() => handleStartLiveness("verify")} style={[styles.modalBtn, styles.confirmBtn, { backgroundColor: colors.primary }]}>
                  <Text style={styles.confirmBtnText}>Start Verification</Text>
                </Pressable>
              </View>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>

      {/* ── Enroll Modal ── */}
      <Modal visible={showEnrollModal} transparent animationType="fade" onRequestClose={() => setShowEnrollModal(false)}>
        <View style={styles.modalBackdrop}>
          <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.modalWrapper}>
            <View style={[styles.modalCard, { backgroundColor: colors.card }]}>
              <View style={styles.modalHeader}>
                <Text style={[styles.modalTitle, { color: colors.foreground }]}>Enroll New Worker</Text>
                <Pressable onPress={() => setShowEnrollModal(false)} style={styles.closeBtn}>
                  <Feather name="x" size={20} color={colors.mutedForeground} />
                </Pressable>
              </View>
              <Text style={[styles.modalDesc, { color: colors.mutedForeground }]}>
                Register a new worker. A facial liveness scan will capture their biometrics.
              </Text>
              <Text style={[styles.fieldLabel, { color: colors.foreground }]}>Employee ID</Text>
              <TextInput
                value={employeeId}
                onChangeText={setEmployeeId}
                placeholder="e.g. NHAI-4029"
                placeholderTextColor={colors.mutedForeground}
                style={[styles.textInput, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background, marginBottom: 12 }]}
                autoCapitalize="characters"
                autoCorrect={false}
                autoFocus
              />
              <Text style={[styles.fieldLabel, { color: colors.foreground }]}>Full Name</Text>
              <TextInput
                value={employeeName}
                onChangeText={setEmployeeName}
                placeholder="e.g. Amit Kumar"
                placeholderTextColor={colors.mutedForeground}
                style={[styles.textInput, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
                autoCapitalize="words"
              />
              <View style={styles.modalButtons}>
                <Pressable onPress={() => setShowEnrollModal(false)} style={[styles.modalBtn, styles.cancelBtn, { borderColor: colors.border }]}>
                  <Text style={[styles.cancelBtnText, { color: colors.foreground }]}>Cancel</Text>
                </Pressable>
                <Pressable onPress={() => handleStartLiveness("enroll")} style={[styles.modalBtn, styles.confirmBtn, { backgroundColor: colors.primary }]}>
                  <Text style={styles.confirmBtnText}>Start Enrollment</Text>
                </Pressable>
              </View>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>

      {/* ── Server Config Modal ── */}
      <Modal visible={showServerModal} transparent animationType="fade" onRequestClose={() => setShowServerModal(false)}>
        <View style={styles.modalBackdrop}>
          <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={styles.modalWrapper}>
            <View style={[styles.modalCard, { backgroundColor: colors.card }]}>
              <View style={styles.modalHeader}>
                <Text style={[styles.modalTitle, { color: colors.foreground }]}>Server Settings</Text>
                <Pressable onPress={() => setShowServerModal(false)} style={styles.closeBtn}>
                  <Feather name="x" size={20} color={colors.mutedForeground} />
                </Pressable>
              </View>
              <Text style={[styles.modalDesc, { color: colors.mutedForeground }]}>
                Set the API server URL. For Android emulator use{" "}
                <Text style={{ fontFamily: "Inter_600SemiBold" }}>10.0.2.2</Text> instead of{" "}
                <Text style={{ fontFamily: "Inter_600SemiBold" }}>localhost</Text>.
              </Text>
              <Text style={[styles.fieldLabel, { color: colors.foreground }]}>API Server URL</Text>
              <TextInput
                value={serverUrlDraft}
                onChangeText={setServerUrlDraft}
                placeholder="http://10.0.2.2:3000"
                placeholderTextColor={colors.mutedForeground}
                style={[styles.textInput, { color: colors.foreground, borderColor: colors.border, backgroundColor: colors.background }]}
                autoCapitalize="none"
                keyboardType="url"
                autoCorrect={false}
              />
              <View style={styles.modalButtons}>
                <Pressable onPress={() => setShowServerModal(false)} style={[styles.modalBtn, styles.cancelBtn, { borderColor: colors.border }]}>
                  <Text style={[styles.cancelBtnText, { color: colors.foreground }]}>Cancel</Text>
                </Pressable>
                <Pressable
                  onPress={() => { setServerUrl(serverUrlDraft.trim()); setShowServerModal(false); }}
                  style={[styles.modalBtn, styles.confirmBtn, { backgroundColor: colors.primary }]}
                >
                  <Text style={styles.confirmBtnText}>Save</Text>
                </Pressable>
              </View>
            </View>
          </KeyboardAvoidingView>
        </View>
      </Modal>

    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root: { flex: 1 },

  // Header
  headerBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingBottom: 12,
    borderBottomWidth: 1,
  },
  headerLeft: { flexDirection: "row", alignItems: "center", gap: 10 },
  logoMini: { width: 32, height: 32, borderRadius: 16, borderWidth: 1, borderColor: "rgba(0,0,0,0.05)" },
  headerTitle: { fontSize: 16, fontFamily: "Inter_700Bold", lineHeight: 18 },
  headerSubtitle: { fontSize: 11, fontFamily: "Inter_500Medium", lineHeight: 12 },
  syncContainerHeader: { flexDirection: "row", alignItems: "center", gap: 6 },
  onlineDot: { width: 8, height: 8, borderRadius: 4 },
  onlineLabel: { fontSize: 11, fontFamily: "Inter_600SemiBold" },
  badgeUnsynced: { borderRadius: 10, paddingHorizontal: 6, paddingVertical: 2, minWidth: 18, alignItems: "center" },
  badgeText: { color: "#fff", fontSize: 10, fontFamily: "Inter_700Bold" },
  syncIconBtn: { width: 36, height: 36, borderRadius: 18, alignItems: "center", justifyContent: "center" },

  // Status bar
  statusBar: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderBottomWidth: 1,
  },
  statusBarText: { fontSize: 12, fontFamily: "Inter_500Medium" },

  // Tabs
  tabContainer: { flexDirection: "row", borderBottomWidth: 1 },
  tabItem: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 13,
    borderBottomWidth: 2,
    borderBottomColor: "transparent",
    gap: 6,
  },
  tabLabel: { fontSize: 12, fontFamily: "Inter_600SemiBold" },

  // Search
  searchContainer: { paddingHorizontal: 16, paddingVertical: 10 },
  searchBox: { flexDirection: "row", alignItems: "center", borderWidth: 1, borderRadius: 10, paddingHorizontal: 10, paddingVertical: 8 },
  searchInput: { flex: 1, fontSize: 14, fontFamily: "Inter_400Regular", padding: 0 },

  // List
  listScroll: { paddingHorizontal: 16, paddingTop: 8 },
  placeholderContainer: { alignItems: "center", justifyContent: "center", paddingVertical: 80, paddingHorizontal: 32 },
  placeholderTitle: { fontSize: 16, fontFamily: "Inter_700Bold", marginTop: 16, marginBottom: 8 },
  placeholderText: { fontSize: 13, fontFamily: "Inter_400Regular", textAlign: "center", lineHeight: 18 },

  sectionHeader: { fontSize: 11, fontFamily: "Inter_600SemiBold", marginBottom: 10, marginTop: 4, textTransform: "uppercase", letterSpacing: 0.8 },

  // Cards
  card: {
    borderRadius: 14,
    borderWidth: 1,
    padding: 14,
    marginBottom: 12,
    elevation: 1,
    shadowColor: "#000",
    shadowOpacity: 0.03,
    shadowOffset: { width: 0, height: 1 },
    shadowRadius: 2,
  },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardHeaderLeft: { flexDirection: "row", alignItems: "center", gap: 10, flex: 1 },
  iconCircle: { width: 32, height: 32, borderRadius: 16, alignItems: "center", justifyContent: "center" },
  cardTitle: { fontSize: 15, fontFamily: "Inter_600SemiBold" },
  cardSubText: { fontSize: 12, fontFamily: "Inter_500Medium", marginTop: 1 },
  cardDivider: { height: 1, backgroundColor: "rgba(0,0,0,0.05)", marginVertical: 10 },
  cardRow: { flexDirection: "row", justifyContent: "space-between" },
  cardCol: { flex: 1 },
  cardLabel: { fontSize: 10, fontFamily: "Inter_500Medium", marginBottom: 2 },
  cardValue: { fontSize: 12, fontFamily: "Inter_600SemiBold" },
  statusPill: { borderRadius: 6, paddingHorizontal: 8, paddingVertical: 3 },
  pillGreen: { backgroundColor: "#dcfce7" },
  pillOrange: { backgroundColor: "#ffedd5" },
  statusPillText: { fontSize: 10, fontFamily: "Inter_700Bold" },
  cardFooter: { marginTop: 10, flexDirection: "row", alignItems: "center", borderRadius: 8, padding: 8 },
  footerText: { fontSize: 10, fontFamily: "Inter_500Medium", flex: 1 },

  // Sync Summary Card
  syncSummaryCard: {
    borderRadius: 16,
    borderWidth: 1,
    padding: 16,
    marginBottom: 16,
  },
  syncSummaryTitle: { fontSize: 14, fontFamily: "Inter_700Bold", marginBottom: 16 },
  syncSummaryRow: { flexDirection: "row", justifyContent: "space-around", alignItems: "center", marginBottom: 16 },
  syncSummaryItem: { alignItems: "center", flex: 1 },
  syncSummaryNum: { fontSize: 28, fontFamily: "Inter_700Bold", lineHeight: 32 },
  syncSummaryLabel: { fontSize: 11, fontFamily: "Inter_500Medium", marginTop: 2 },
  syncSummaryDivider: { width: 1, height: 40 },
  syncNowBtn: { flexDirection: "row", alignItems: "center", justifyContent: "center", borderRadius: 12, paddingVertical: 13 },
  syncNowBtnText: { color: "#fff", fontSize: 15, fontFamily: "Inter_700Bold" },

  // FAB
  fabContainer: { position: "absolute", left: 24, right: 24, elevation: 8, shadowColor: "#000", shadowOpacity: 0.15, shadowOffset: { width: 0, height: 4 }, shadowRadius: 6 },
  primaryFab: { flexDirection: "row", alignItems: "center", justifyContent: "center", borderRadius: 16, paddingVertical: 14 },
  primaryFabText: { color: "#fff", fontSize: 15, fontFamily: "Inter_700Bold" },

  // Modals
  modalBackdrop: { flex: 1, backgroundColor: "rgba(0,0,0,0.55)", justifyContent: "center", alignItems: "center", padding: 24 },
  modalWrapper: { width: "100%", alignItems: "center" },
  modalCard: { width: "100%", borderRadius: 20, padding: 20, elevation: 10, shadowColor: "#000", shadowOpacity: 0.25, shadowOffset: { width: 0, height: 5 }, shadowRadius: 10 },
  modalHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 10 },
  modalTitle: { fontSize: 18, fontFamily: "Inter_700Bold" },
  closeBtn: { width: 32, height: 32, borderRadius: 16, alignItems: "center", justifyContent: "center" },
  modalDesc: { fontSize: 13, fontFamily: "Inter_400Regular", lineHeight: 18, marginBottom: 20 },
  fieldLabel: { fontSize: 12, fontFamily: "Inter_600SemiBold", marginBottom: 6 },
  textInput: { borderRadius: 8, borderWidth: 1, paddingHorizontal: 12, paddingVertical: 10, fontSize: 15, fontFamily: "Inter_400Regular" },
  modalButtons: { flexDirection: "row", gap: 12, marginTop: 20 },
  modalBtn: { flex: 1, paddingVertical: 12, borderRadius: 10, alignItems: "center", justifyContent: "center" },
  cancelBtn: { borderWidth: 1 },
  cancelBtnText: { fontSize: 14, fontFamily: "Inter_600SemiBold" },
  confirmBtn: { elevation: 2 },
  confirmBtnText: { color: "#fff", fontSize: 14, fontFamily: "Inter_700Bold" },
}) as Record<string, any>;
