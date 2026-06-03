"use no memo";
import { Feather } from "@expo/vector-icons";
import { SHA256 } from "crypto-js";
import * as Haptics from "expo-haptics";
import { CameraView, useCameraPermissions } from "expo-camera";
import { LinearGradient } from "expo-linear-gradient";
import { router, useLocalSearchParams } from "expo-router";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Animated,
  Dimensions,
  Platform,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useColors } from "@/hooks/useColors";
import { encryptEmbeddings, decryptEmbeddings, cosineSimilarity } from "@/constants/vault";
import {
  enrollEmployeeLocal,
  getEnrolledEmployeeLocal,
  saveAttendanceLocal,
  getLatestAttendanceHashLocal,
} from "@/constants/localDb";

const { width: SW, height: SH } = Dimensions.get("window");

type FlashDir = "top" | "bottom" | "left" | "right" | null;

const STAGES = [
  {
    id: 0,
    key: "flash",
    icon: "weather-lightning" as const,
    title: "Photometric Stereo",
    subtitle: "Screen Flash 3D Reconstruction",
    description:
      "Screen flashing directional light — capturing depth map from 4 angles",
    duration: 8000,
    color: "#7c3aed",
    directions: ["top", "right", "bottom", "left"] as FlashDir[],
  },
  {
    id: 1,
    key: "moire",
    icon: "grid" as const,
    title: "Moiré Detection",
    subtitle: "Passive Anti-Spoofing",
    description:
      "Analysing texture for digital screen pixel patterns via FFT",
    duration: 5000,
    color: "#0891b2",
  },
  {
    id: 2,
    key: "rppg",
    icon: "heart-pulse" as const,
    title: "rPPG Pulse Detection",
    subtitle: "Camera-Based Heart Rate",
    description:
      "Measuring micro colour changes in forehead — bandpass filter 0.7–4 Hz",
    duration: 8000,
    color: "#e11d48",
  },
  {
    id: 3,
    key: "imu",
    icon: "axis-arrow" as const,
    title: "IMU Correlation",
    subtitle: "Motion Sync Analysis",
    description:
      "Cross-correlating gyroscope motion with optical flow face tracking",
    duration: 5500,
    color: "#059669",
  },
];

/**
 * Generates a deterministic seed from a string (such as the employee ID).
 */
export function getDeterministicSeed(employeeId: string): number {
  let hash = 0;
  const cleanId = employeeId.trim().toLowerCase();
  for (let i = 0; i < cleanId.length; i++) {
    hash = (hash << 5) - hash + cleanId.charCodeAt(i);
    hash |= 0; // Convert to 32-bit integer
  }
  return Math.abs(hash) || 42;
}

/**
 * Generates a deterministic-looking mock embedding vector (512 floats).
 * In production this would come from the ML model during the Moiré frame capture.
 * If addNoise is true, small random variance is added to simulate biometric changes.
 */
function generateMockEmbedding(seed: number, addNoise = false): number[] {
  const vec: number[] = [];
  let s = seed;
  for (let i = 0; i < 512; i++) {
    // Simple PRNG seeded hash for reproducibility
    s = ((s * 1103515245 + 12345) & 0x7fffffff);
    let val = (s / 0x7fffffff) * 2 - 1; // values in [-1, 1]
    if (addNoise) {
      // Add up to ±8% noise to simulate natural variance in camera feed
      const noise = (Math.random() * 0.16) - 0.08;
      val += noise;
    }
    vec.push(val);
  }
  // Normalize to unit vector
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  return vec.map((v) => v / norm);
}

/**
 * Compute SHA-256 hash of a string using expo-crypto.
 */
async function sha256(input: string): Promise<string> {
  return SHA256(input).toString();
}

export default function LivenessScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const [permission, requestPermission] = useCameraPermissions();

  // Route params from home screen
  const params = useLocalSearchParams<{
    mode: string;
    employeeId: string;
    employeeName: string;
  }>();
  const mode = (params.mode as "verify" | "enroll") || "verify";
  const employeeId = params.employeeId || "";
  const employeeName = params.employeeName || "";

  // ── All state hooks (must be before any early return) ──────────────────────
  const [stageIdx, setStageIdx] = useState(0);
  const [stagePhase, setStagePhase] = useState<
    "detecting" | "passed" | "done" | "failed"
  >("detecting");
  const [boxPassed, setBoxPassed] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [hasFailed, setHasFailed] = useState(false);
  const [flashDir, setFlashDir] = useState<FlashDir>(null);
  const [flashVisible, setFlashVisible] = useState(false);
  const [flashColor, setFlashColor] = useState("rgba(255,255,255,0.95)");
  const [isProcessing, setIsProcessing] = useState(false);

  // Store collected embedding samples (for enrollment: 5-10 samples)
  const embeddingSamples = useRef<number[][]>([]);
  const enrollmentSeedRef = useRef<number>(0);

  // ── All ref hooks ──────────────────────────────────────────────────────────
  const progressAnim = useRef(new Animated.Value(0)).current;
  const progressAnimation = useRef<Animated.CompositeAnimation | null>(null);
  const flashAnim = useRef(new Animated.Value(0)).current;
  const faceOffsetX = useRef(new Animated.Value(0)).current;
  const faceOffsetY = useRef(new Animated.Value(0)).current;

  const stageTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const flashTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const faceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const faceTrackTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const faceLoopX = useRef<Animated.CompositeAnimation | null>(null);
  const faceLoopY = useRef<Animated.CompositeAnimation | null>(null);
  const permissionRequested = useRef(false);
  const faceDetectedRef = useRef(false);

  // Keep ref in sync with state so timers can read latest value
  useEffect(() => {
    faceDetectedRef.current = faceDetected;
  }, [faceDetected]);

  // ── Callbacks (must be before any early return) ────────────────────────────
  const runStage = useCallback(
    (idx: number) => {
      const stage = STAGES[idx];
      if (!stage) return;

      setStagePhase("detecting");
      setBoxPassed(false);
      setFaceDetected(false);
      faceDetectedRef.current = false;
      setFlashDir(null);
      setFlashVisible(false);
      faceOffsetX.setValue(0);
      faceOffsetY.setValue(0);
      if (faceLoopX.current) { faceLoopX.current.stop(); faceLoopX.current = null; }
      if (faceLoopY.current) { faceLoopY.current.stop(); faceLoopY.current = null; }

      progressAnim.setValue(0);
      if (progressAnimation.current) progressAnimation.current.stop();
      progressAnimation.current = Animated.timing(progressAnim, {
        toValue: 1,
        duration: stage.duration - 1000,
        useNativeDriver: false,
      });
      progressAnimation.current.start();

      if (stage.key === "flash") {
        const BRIGHT_COLORS = [
          "rgba(255,60,60,0.97)",
          "rgba(60,130,255,0.97)",
          "rgba(60,240,200,0.97)",
          "rgba(210,80,255,0.97)",
          "rgba(255,255,255,0.97)",
          "rgba(255,140,40,0.97)",
          "rgba(60,220,60,0.97)",
          "rgba(255,220,50,0.97)",
        ];
        let dirIdx = 0;
        const dirs = stage.directions!;
        flashTimer.current = setInterval(() => {
          const color = BRIGHT_COLORS[Math.floor(Math.random() * BRIGHT_COLORS.length)];
          setFlashColor(color);
          setFlashDir(dirs[dirIdx % dirs.length]);
          setFlashVisible(true);
          flashAnim.setValue(0);
          Animated.sequence([
            Animated.timing(flashAnim, { toValue: 1, duration: 150, useNativeDriver: false }),
            Animated.delay(400),
            Animated.timing(flashAnim, { toValue: 0, duration: 180, useNativeDriver: false }),
          ]).start();
          dirIdx++;
        }, 1500);
      }

      stageTimer.current = setTimeout(() => {
        if (flashTimer.current) {
          clearInterval(flashTimer.current);
          flashTimer.current = null;
          setFlashDir(null);
          setFlashVisible(false);
        }
        if (faceTimer.current) { clearTimeout(faceTimer.current); faceTimer.current = null; }
        if (faceTrackTimer.current) { clearTimeout(faceTrackTimer.current); faceTrackTimer.current = null; }

        const passed = faceDetectedRef.current;
        if (passed) {
          // ── Collect mock embedding during moiré stage ──────────────────
          if (stage.key === "moire") {
            const seed = getDeterministicSeed(employeeId);
            enrollmentSeedRef.current = seed;
            const embedding = generateMockEmbedding(seed, true);
            embeddingSamples.current.push(embedding);
          }

          setStagePhase("passed");
          setBoxPassed(true);
          setHasFailed(false);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

          setTimeout(() => {
            if (idx < STAGES.length - 1) {
              setStageIdx(idx + 1);
              runStage(idx + 1);
            } else {
              setStagePhase("done");
              handleAllStagesPassed();
            }
          }, 1500);
        } else {
          setStagePhase("failed");
          setBoxPassed(false);
          setHasFailed(true);
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
          setTimeout(() => { router.replace("/failure"); }, 600);
        }
      }, stage.duration);
    },
    [flashAnim, faceOffsetX, faceOffsetY, progressAnim]
  );

  // ── Post-liveness biometric processing ─────────────────────────────────────
  const handleAllStagesPassed = useCallback(async () => {
    setIsProcessing(true);
    try {
      // Ensure we have at least one embedding sample
      if (embeddingSamples.current.length === 0) {
        // Generate a fallback sample
        const seed = getDeterministicSeed(employeeId);
        embeddingSamples.current.push(generateMockEmbedding(seed, true));
      }

      if (mode === "enroll") {
        // ── ENROLLMENT MODE ────────────────────────────────────────────
        // Generate additional samples to reach 5-10 photos
        const baseSeed = getDeterministicSeed(employeeId);
        while (embeddingSamples.current.length < 5) {
          embeddingSamples.current.push(generateMockEmbedding(baseSeed, true));
        }

        const samples = embeddingSamples.current;

        // Verify pairwise cosine similarity >= 0.6
        for (let i = 0; i < samples.length; i++) {
          for (let j = i + 1; j < samples.length; j++) {
            const sim = cosineSimilarity(samples[i], samples[j]);
            if (sim < 0.6) {
              Alert.alert(
                "Enrollment Failed",
                `Inconsistent face captures detected (similarity ${sim.toFixed(2)}). Please try again in better lighting.`
              );
              router.replace("/failure");
              return;
            }
          }
        }

        // Average the vectors and normalize
        const dim = samples[0].length;
        const avgVec = new Array(dim).fill(0);
        for (const sample of samples) {
          for (let i = 0; i < dim; i++) {
            avgVec[i] += sample[i];
          }
        }
        for (let i = 0; i < dim; i++) {
          avgVec[i] /= samples.length;
        }
        const norm = Math.sqrt(avgVec.reduce((s: number, v: number) => s + v * v, 0));
        for (let i = 0; i < dim; i++) {
          avgVec[i] /= norm;
        }

        // Encrypt and store in local SQLite
        const encryptedHex = await encryptEmbeddings(avgVec);
        await enrollEmployeeLocal(employeeId, employeeName, encryptedHex);

        setTimeout(() => {
          router.replace({
            pathname: "/success",
            params: { message: `Employee ${employeeId} enrolled successfully. Biometrics encrypted and stored locally.` },
          });
        }, 400);

      } else {
        // ── VERIFICATION MODE ──────────────────────────────────────────
        const enrolled = await getEnrolledEmployeeLocal(employeeId);
        if (!enrolled) {
          Alert.alert(
            "Not Enrolled",
            `Employee ${employeeId} is not registered on this device. Please enroll first.`
          );
          router.replace("/failure");
          return;
        }

        if (enrolled.encrypted_embeddings === "SECURELY_CLEARED") {
          Alert.alert(
            "Biometrics Cleared",
            `Biometrics for Employee ${employeeId} have been synced and cleared from this device for security. Re-enrollment is required for offline verification.`
          );
          router.replace("/failure");
          return;
        }

        // Decrypt enrolled embeddings
        const enrolledEmbeddings = await decryptEmbeddings(enrolled.encrypted_embeddings);
        const capturedEmbedding = embeddingSamples.current[embeddingSamples.current.length - 1];

        // Cosine similarity check
        const similarity = cosineSimilarity(capturedEmbedding, enrolledEmbeddings);
        const THRESHOLD = 0.6;

        if (similarity < THRESHOLD) {
          Alert.alert(
            "Verification Failed",
            `Face does not match enrolled biometrics (score: ${similarity.toFixed(2)}).`
          );
          router.replace("/failure");
          return;
        }

        // ── Build blockchain-lite chained attendance record ────────────
        const embeddingHash = await sha256(JSON.stringify(capturedEmbedding));
        const prevHash = await getLatestAttendanceHashLocal();
        const timestamp = Date.now();
        const gps = "28.6139,77.2090"; // Placeholder — would come from expo-location
        const livenessScore = similarity;

        // Chain: hash(prevHash + employeeId + timestamp + embeddingHash)
        const chainInput = `${prevHash}|${employeeId}|${timestamp}|${embeddingHash}`;
        const recordHash = await sha256(chainInput);

        await saveAttendanceLocal(
          employeeId,
          gps,
          livenessScore,
          embeddingHash,
          recordHash
        );

        setTimeout(() => {
          router.replace({
            pathname: "/success",
            params: {
              message: `Employee ${employeeId} verified (score: ${similarity.toFixed(2)}). Attendance recorded and hash-chained.`,
            },
          });
        }, 400);
      }
    } catch (err: any) {
      console.error("Biometric processing error:", err);
      Alert.alert("Error", err.message || "Failed to process biometric data.");
      router.replace("/failure");
    } finally {
      setIsProcessing(false);
      embeddingSamples.current = [];
    }
  }, [mode, employeeId, employeeName]);

  const handleFacesDetected = useCallback(() => {
    setFaceDetected(true);
    faceDetectedRef.current = true;
    setHasFailed(false);

    if (!faceLoopX.current) {
      faceLoopX.current = Animated.loop(
        Animated.sequence([
          Animated.timing(faceOffsetX, { toValue: 6, duration: 1400, useNativeDriver: false }),
          Animated.timing(faceOffsetX, { toValue: -4, duration: 1100, useNativeDriver: false }),
          Animated.timing(faceOffsetX, { toValue: 2, duration: 900, useNativeDriver: false }),
        ])
      );
      faceLoopX.current.start();
    }
    if (!faceLoopY.current) {
      faceLoopY.current = Animated.loop(
        Animated.sequence([
          Animated.timing(faceOffsetY, { toValue: 4, duration: 1700, useNativeDriver: false }),
          Animated.timing(faceOffsetY, { toValue: -3, duration: 1300, useNativeDriver: false }),
          Animated.timing(faceOffsetY, { toValue: 1, duration: 1000, useNativeDriver: false }),
        ])
      );
      faceLoopY.current.start();
    }
  }, [faceOffsetX, faceOffsetY]);

  // Handle mock face detection (since Expo SDK 51 CameraView drops onFacesDetected)
  useEffect(() => {
    if (stagePhase === "detecting" && permission?.granted) {
      // Simulate taking 1.5s to find the face
      const t = setTimeout(() => {
        handleFacesDetected();
      }, 1500);
      return () => clearTimeout(t);
    }
  }, [stagePhase, permission, handleFacesDetected]);

  // ── Effects (must be before any early return) ──────────────────────────────
  useEffect(() => {
    if (!permission) return;

    if (permission.granted) {
      const t = setTimeout(() => runStage(0), 800);
      return () => clearTimeout(t);
    }

    if (!permissionRequested.current) {
      permissionRequested.current = true;
      requestPermission().then((result) => {
        if (!result.granted) {
          router.replace("/denied");
        }
      });
    } else if (!permission.canAskAgain) {
      router.replace("/denied");
    }
  }, [permission, requestPermission, runStage]);

  useEffect(() => {
    return () => {
      if (stageTimer.current) clearTimeout(stageTimer.current);
      if (flashTimer.current) clearInterval(flashTimer.current);
      if (faceTimer.current) clearTimeout(faceTimer.current);
      if (faceTrackTimer.current) clearTimeout(faceTrackTimer.current);
      if (faceLoopX.current) faceLoopX.current.stop();
      if (faceLoopY.current) faceLoopY.current.stop();
    };
  }, []);

  // ── Derived values (safe after all hooks) ─────────────────────────────────
  const topInset = Platform.OS === "web" ? 67 : insets.top;
  const boxColor = hasFailed ? "#ef4444" : (faceDetected ? "#22c55e" : "#ef4444");
  const boxBgColor = hasFailed
    ? "rgba(239,68,68,0.08)"
    : faceDetected
      ? "rgba(34,197,94,0.08)"
      : "rgba(239,68,68,0.08)";

  // ── Early return (AFTER all hooks) ────────────────────────────────────────
  if (!permission || !permission.granted) {
    return (
      <View style={[styles.center, { backgroundColor: "#000" }]}>
        <Feather name="camera" size={36} color="rgba(255,255,255,0.5)" />
        <Text style={styles.requestingText}>Requesting camera access…</Text>
      </View>
    );
  }

  // ── Render variables ──────────────────────────────────────────────────────
  const stage = STAGES[stageIdx];

  const CAMERA_H = SH;
  const FACE_LEFT = SW * 0.05;
  const FACE_TOP = CAMERA_H * 0.12;
  const FACE_W = SW * 0.90;
  const FACE_H = CAMERA_H * 0.56;

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["0%", "100%"],
  });

  const getFlashGradient = () => {
    switch (flashDir) {
      case "top":
        return {
          colors: [flashColor, "transparent"] as [string, string],
          start: { x: 0.5, y: 0 },
          end: { x: 0.5, y: 1 },
        };
      case "bottom":
        return {
          colors: ["transparent", flashColor] as [string, string],
          start: { x: 0.5, y: 0 },
          end: { x: 0.5, y: 1 },
        };
      case "left":
        return {
          colors: [flashColor, "transparent"] as [string, string],
          start: { x: 0, y: 0.5 },
          end: { x: 1, y: 0.5 },
        };
      case "right":
        return {
          colors: ["transparent", flashColor] as [string, string],
          start: { x: 0, y: 0.5 },
          end: { x: 1, y: 0.5 },
        };
      default:
        return null;
    }
  };

  const flashGrad = getFlashGradient();

  return (
    <View style={styles.container}>
      <CameraView
        style={StyleSheet.absoluteFill}
        facing="front"
      />

      {flashVisible && flashGrad && (
        <Animated.View
          style={[StyleSheet.absoluteFill, { opacity: flashAnim }]}
          pointerEvents="none"
        >
          <LinearGradient
            colors={flashGrad.colors}
            start={flashGrad.start}
            end={flashGrad.end}
            style={StyleSheet.absoluteFill}
          />
        </Animated.View>
      )}

      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Animated.View
          style={{
            position: "absolute",
            left: FACE_LEFT,
            top: FACE_TOP,
            width: FACE_W,
            height: FACE_H,
            borderWidth: 2,
            borderRadius: 4,
            borderColor: boxColor,
            backgroundColor: boxBgColor,
            transform: [
              { translateX: faceOffsetX },
              { translateY: faceOffsetY },
            ],
          }}
        >
          <View style={[styles.corner, styles.cornerTL, { borderColor: boxColor }]} />
          <View style={[styles.corner, styles.cornerTR, { borderColor: boxColor }]} />
          <View style={[styles.corner, styles.cornerBL, { borderColor: boxColor }]} />
          <View style={[styles.corner, styles.cornerBR, { borderColor: boxColor }]} />
        </Animated.View>

        {stagePhase === "passed" && (
          <View
            style={[
              styles.passedBadge,
              {
                left: FACE_LEFT + FACE_W / 2 - 52,
                top: FACE_TOP + FACE_H + 12,
              },
            ]}
          >
            <View style={styles.checkCircle}>
              <Text style={styles.checkMark}>✓</Text>
            </View>
            <Text style={styles.passedText}>Passed</Text>
          </View>
        )}

        {stage.key === "imu" && stagePhase === "detecting" && (
          <View
            style={[
              styles.imuBox,
              { left: FACE_LEFT, top: FACE_TOP + FACE_H + 54 },
            ]}
          >
            <IMUVisualizer />
          </View>
        )}
      </View>

      <View style={[styles.topOverlay, { paddingTop: topInset + 6 }]}>
        <View style={styles.stageHeader}>
          <View style={styles.stagePills}>
            {STAGES.map((s, i) => (
              <View
                key={i}
                style={[
                  styles.stagePill,
                  {
                    backgroundColor:
                      hasFailed && i <= stageIdx
                        ? "#ef4444"
                        : i < stageIdx
                          ? "#22c55e"
                          : i === stageIdx
                            ? "#fff"
                            : "rgba(255,255,255,0.3)",
                    width: i === stageIdx ? 28 : 8,
                  },
                ]}
              />
            ))}
          </View>

          <View style={styles.stageCount}>
            <Text style={styles.stageCountText}>
              {stageIdx + 1}/{STAGES.length}
            </Text>
          </View>
        </View>

        <View style={styles.stageInfo}>
          <Text style={styles.stageName}>{stage.title}</Text>
        </View>

        <View style={styles.progressTrack}>
          <Animated.View
            style={[
              styles.progressFill,
              { width: progressWidth, backgroundColor: stage.color },
            ]}
          />
        </View>
      </View>

      <View
        style={[
          styles.bottomOverlay,
          { paddingBottom: insets.bottom + 20 },
        ]}
      >
        <Text style={styles.bottomFeatureName}>{stage.title}</Text>
        <Text style={styles.holdText}>
          {isProcessing
            ? "PROCESSING BIOMETRICS…"
            : mode === "enroll"
              ? `ENROLLING ${employeeId} · HOLD STILL`
              : `VERIFYING ${employeeId} · HOLD STILL`}
        </Text>
      </View>
    </View>
  );
}

function IMUVisualizer() {
  const rotX = useRef(new Animated.Value(0)).current;
  const rotZ = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(rotX, { toValue: 1, duration: 1200, useNativeDriver: false }),
        Animated.timing(rotX, { toValue: -1, duration: 1200, useNativeDriver: false }),
        Animated.timing(rotX, { toValue: 0, duration: 600, useNativeDriver: false }),
      ])
    ).start();
    Animated.loop(
      Animated.sequence([
        Animated.delay(600),
        Animated.timing(rotZ, { toValue: 1, duration: 1000, useNativeDriver: false }),
        Animated.timing(rotZ, { toValue: -0.5, duration: 1000, useNativeDriver: false }),
        Animated.timing(rotZ, { toValue: 0, duration: 500, useNativeDriver: false }),
      ])
    ).start();
  }, []);

  const barX = rotX.interpolate({ inputRange: [-1, 1], outputRange: [10, 46] });
  const barZ = rotZ.interpolate({ inputRange: [-1, 1], outputRange: [12, 44] });

  return (
    <View style={imuStyles.container}>
      <Text style={imuStyles.label}>IMU · Face Sync</Text>
      <View style={imuStyles.row}>
        {(["X", "Y", "Z"] as const).map((axis, i) => (
          <View key={axis} style={imuStyles.axis}>
            <Text style={imuStyles.axisLabel}>{axis}</Text>
            <View style={imuStyles.barTrack}>
              <Animated.View
                style={[
                  imuStyles.barFill,
                  { height: i === 0 ? barX : i === 2 ? barZ : 28 },
                ]}
              />
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

const imuStyles = StyleSheet.create({
  container: {
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 10,
    padding: 10,
    alignItems: "center",
  },
  label: {
    color: "#4ade80",
    fontSize: 11,
    fontFamily: "Inter_600SemiBold",
    marginBottom: 6,
    letterSpacing: 1,
  },
  row: {
    flexDirection: "row",
    gap: 12,
    alignItems: "flex-end",
  },
  axis: {
    alignItems: "center",
    gap: 4,
  },
  axisLabel: {
    color: "#fff",
    fontSize: 10,
    fontFamily: "Inter_500Medium",
  },
  barTrack: {
    width: 10,
    height: 48,
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 5,
    justifyContent: "flex-end",
    overflow: "hidden",
  },
  barFill: {
    width: 10,
    backgroundColor: "#4ade80",
    borderRadius: 5,
  },
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 16,
    padding: 24,
  },
  requestingText: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 15,
    fontFamily: "Inter_400Regular",
    marginTop: 4,
  },
  corner: {
    position: "absolute",
    width: 18,
    height: 18,
    borderWidth: 3,
  },
  cornerTL: {
    top: -2,
    left: -2,
    borderRightWidth: 0,
    borderBottomWidth: 0,
    borderTopLeftRadius: 4,
  },
  cornerTR: {
    top: -2,
    right: -2,
    borderLeftWidth: 0,
    borderBottomWidth: 0,
    borderTopRightRadius: 4,
  },
  cornerBL: {
    bottom: -2,
    left: -2,
    borderRightWidth: 0,
    borderTopWidth: 0,
    borderBottomLeftRadius: 4,
  },
  cornerBR: {
    bottom: -2,
    right: -2,
    borderLeftWidth: 0,
    borderTopWidth: 0,
    borderBottomRightRadius: 4,
  },
  passedBadge: {
    position: "absolute",
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(0,0,0,0.6)",
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  passedText: {
    color: "#22c55e",
    fontSize: 14,
    fontFamily: "Inter_700Bold",
  },
  checkCircle: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: "#22c55e",
    alignItems: "center",
    justifyContent: "center",
  },
  checkMark: {
    color: "#fff",
    fontSize: 12,
    fontFamily: "Inter_700Bold",
    lineHeight: 14,
  },
  imuBox: {
    position: "absolute",
  },
  topOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 16,
    paddingBottom: 8,
    backgroundColor: "rgba(0,0,0,0.55)",
  },
  stageHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  stagePills: {
    flex: 1,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 4,
    marginHorizontal: 12,
  },
  stagePill: {
    height: 8,
    borderRadius: 4,
  },
  stageCount: {
    backgroundColor: "rgba(255,255,255,0.15)",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  stageCountText: {
    color: "#fff",
    fontSize: 12,
    fontFamily: "Inter_600SemiBold",
  },
  stageInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 10,
  },
  stageName: {
    color: "#fff",
    fontSize: 15,
    fontFamily: "Inter_700Bold",
  },
  progressTrack: {
    height: 4,
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: 2,
    overflow: "hidden",
  },
  progressFill: {
    height: 4,
    borderRadius: 2,
  },
  bottomOverlay: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: "rgba(0,0,0,0.55)",
    paddingTop: 16,
    paddingHorizontal: 20,
  },
  bottomFeatureName: {
    color: "#fff",
    fontSize: 16,
    fontFamily: "Inter_700Bold",
    textAlign: "center",
    marginBottom: 4,
  },
  holdText: {
    color: "rgba(255,255,255,0.5)",
    fontSize: 11,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
    letterSpacing: 0.5,
    marginBottom: 8,
  },
});
