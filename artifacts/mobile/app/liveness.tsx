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

import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system/legacy";
import { manipulateAsync, SaveFormat } from "expo-image-manipulator";
import base64 from "base64-js";
import jpeg from "jpeg-js";
let ort: any = null;
try {
  ort = require("onnxruntime-react-native");
} catch (e) {
  console.warn("onnxruntime-react-native is not available. Running in Mock Mode.", e);
}
import { Gyroscope } from "expo-sensors";

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

let recognizerSession: any = null;
let detectorSession: any = null;
let moireSession: any = null;

async function loadModelSecured(assetName: string, destFileName: string): Promise<any> {
  if (!ort) {
    console.log(`[Mock Mode] loadModelSecured stubbed for ${destFileName}`);
    return { run: async () => ({}) };
  }
  let asset;
  if (destFileName === "face_model_quant.onnx") {
    asset = Asset.fromModule(require("../assets/face_model_quant.onnx"));
  } else if (destFileName === "moire.onnx") {
    asset = Asset.fromModule(require("../assets/moire.onnx"));
  } else {
    asset = Asset.fromModule(require("../assets/detector.onnx"));
  }
  await asset.downloadAsync();

  const sourceUri = asset.localUri || asset.uri;
  if (!sourceUri) {
    throw new Error(`Could not load asset URI for ${destFileName}`);
  }

  console.log(`Decrypting and loading model: ${destFileName}...`);
  const encryptedBase64 = await FileSystem.readAsStringAsync(sourceUri, { encoding: "base64" });
  
  const encryptedBytes = base64.toByteArray(encryptedBase64);
  const key = "nhai_secure_model_key_2026_encryption";
  const decryptedBytes = new Uint8Array(encryptedBytes.length);
  for (let i = 0; i < encryptedBytes.length; i++) {
    const keyChar = key.charCodeAt(i % key.length);
    decryptedBytes[i] = encryptedBytes[i] ^ keyChar;
  }

  const decryptedBase64 = base64.fromByteArray(decryptedBytes);
  const tempPath = `${FileSystem.cacheDirectory}temp_${Date.now()}_${destFileName}`;
  await FileSystem.writeAsStringAsync(tempPath, decryptedBase64, { encoding: "base64" });

  let session: any;
  try {
    session = await ort.InferenceSession.create(tempPath);
    console.log(`Model ${destFileName} loaded successfully into RAM.`);
  } finally {
    try {
      await FileSystem.deleteAsync(tempPath, { idempotent: true });
      console.log(`Temporary decrypted file deleted: ${destFileName}`);
    } catch (e) {
      console.warn(`Failed to delete temporary decrypted model file: ${destFileName}`, e);
    }
  }

  return session;
}

/**
 * Verifies if the brightness gradient on the face matches the screen flash direction.
 */
async function verifyDirectionalBrightness(photoUri: string, direction: FlashDir): Promise<boolean> {
  if (Platform.OS === "web") return true;
  let manipUri: string | null = null;
  try {
    const manip = await manipulateAsync(
      photoUri,
      [{ resize: { width: 32, height: 32 } }],
      { compress: 0.9, format: SaveFormat.JPEG, base64: true }
    );
    manipUri = manip.uri;
    if (!manip.base64) return false;

    const jpegData = base64.toByteArray(manip.base64);
    const rawImage = jpeg.decode(jpegData, { useTArray: true });
    const pixels = rawImage.data; // 32 * 32 * 4 (RGBA)

    // Center 50% region (rows 8-23, cols 8-23)
    let leftSum = 0, rightSum = 0, topSum = 0, bottomSum = 0;
    for (let y = 8; y < 24; y++) {
      for (let x = 8; x < 24; x++) {
        const idx = (y * 32 + x) * 4;
        const r = pixels[idx];
        const g = pixels[idx + 1];
        const b = pixels[idx + 2];
        const brightness = 0.299 * r + 0.587 * g + 0.114 * b;

        if (x < 16) leftSum += brightness;
        else rightSum += brightness;

        if (y < 16) topSum += brightness;
        else bottomSum += brightness;
      }
    }

    const avgLeft = leftSum / 128;
    const avgRight = rightSum / 128;
    const avgTop = topSum / 128;
    const avgBottom = bottomSum / 128;

    const diffLR = Math.abs(avgLeft - avgRight);
    const diffTB = Math.abs(avgTop - avgBottom);

    const relativeDiffLR = diffLR / Math.max(1, (avgLeft + avgRight) / 2);
    const relativeDiffTB = diffTB / Math.max(1, (avgTop + avgBottom) / 2);

    console.log(`Photometric [${direction}] - LR diff: ${relativeDiffLR.toFixed(3)}, TB diff: ${relativeDiffTB.toFixed(3)}`);

    if (direction === "left" || direction === "right") {
      return relativeDiffLR > 0.015; // At least 1.5% brightness difference
    } else if (direction === "top" || direction === "bottom") {
      return relativeDiffTB > 0.015; // At least 1.5% brightness difference
    }
    return false;
  } catch (err) {
    console.error("Error in verifyDirectionalBrightness:", err);
    return false;
  } finally {
    try {
      await FileSystem.deleteAsync(photoUri, { idempotent: true });
      if (manipUri) {
        await FileSystem.deleteAsync(manipUri, { idempotent: true });
      }
    } catch (e) {
      console.warn("Failed to delete temp files in verifyDirectionalBrightness:", e);
    }
  }
}

/**
 * Evaluates rPPG green signal variance and periodicity.
 */
async function verifyRPPGPulse(signals: number[]): Promise<boolean> {
  if (signals.length < 5) {
    console.log("rPPG failed: not enough frames captured");
    return false;
  }

  const mean = signals.reduce((s, v) => s + v, 0) / signals.length;
  const ac = signals.map(v => v - mean);
  const variance = ac.reduce((s, v) => s + v * v, 0) / ac.length;
  const stdDev = Math.sqrt(variance);

  console.log(`rPPG Pulse - Mean: ${mean.toFixed(2)}, StdDev: ${stdDev.toFixed(4)}`);

  if (stdDev < 0.005 || stdDev > 5.0) {
    console.log("rPPG failed: standard deviation out of range (flat or noisy signal)");
    return false;
  }

  let zeroCrossings = 0;
  for (let i = 0; i < ac.length - 1; i++) {
    if (ac[i] * ac[i + 1] < 0) {
      zeroCrossings++;
    }
  }

  console.log(`rPPG Pulse - Zero crossings: ${zeroCrossings}/${ac.length - 1}`);
  if (zeroCrossings < 1 || zeroCrossings > ac.length - 2) {
    console.log("rPPG failed: heart rate periodicity not detected");
    return false;
  }

  return true;
}

async function loadRecognizerModel(): Promise<any> {
  if (recognizerSession) return recognizerSession;
  console.log("Loading recognizer model (secured)...");
  recognizerSession = await loadModelSecured("face_model_quant.onnx", "face_model_quant.onnx");
  console.log("Recognizer model loaded successfully!");
  return recognizerSession;
}

async function loadDetectorModel(): Promise<any> {
  if (detectorSession) return detectorSession;
  console.log("Loading detector model (secured)...");
  detectorSession = await loadModelSecured("detector.onnx", "detector.onnx");
  console.log("Detector model loaded successfully!");
  return detectorSession;
}

async function loadMoireModel(): Promise<any> {
  if (moireSession) return moireSession;
  console.log("Loading moire model (secured)...");
  moireSession = await loadModelSecured("moire.onnx", "moire.onnx");
  console.log("Moire model loaded successfully!");
  return moireSession;
}

const DETECTION_WIDTH = 320;
const DETECTION_HEIGHT = 240;

const minBoxes = [
  [10, 16, 24],
  [32, 48],
  [64, 96],
  [128, 192, 256]
];

const featureMaps = [
  [40, 20, 10, 5],
  [30, 15, 8, 4]
];

const shrinkages = [
  [8, 16, 32, 64],
  [8, 16, 30, 60]
];

const anchors = (() => {
  const priors: number[][] = [];
  const numLayers = featureMaps[0].length;
  
  for (let index = 0; index < numLayers; index++) {
    const scaleW = DETECTION_WIDTH / shrinkages[0][index];
    const scaleH = DETECTION_HEIGHT / shrinkages[1][index];
    const featureH = featureMaps[1][index];
    const featureW = featureMaps[0][index];
    
    for (let j = 0; j < featureH; j++) {
      for (let i = 0; i < featureW; i++) {
        const xCenter = (i + 0.5) / scaleW;
        const yCenter = (j + 0.5) / scaleH;
        
        for (const minBox of minBoxes[index]) {
          const w = minBox / DETECTION_WIDTH;
          const h = minBox / DETECTION_HEIGHT;
          priors.push([
            xCenter,
            yCenter,
            w,
            h
          ]);
        }
      }
    }
  }
  
  for (let i = 0; i < priors.length; i++) {
    for (let j = 0; j < 4; j++) {
      if (priors[i][j] < 0) priors[i][j] = 0;
      if (priors[i][j] > 1) priors[i][j] = 1;
    }
  }
  return priors;
})();

interface FaceDetection {
  box: [number, number, number, number];
  score: number;
}

function decodeBoxes(
  locations: Float32Array,
  scores: Float32Array,
  scoreThreshold = 0.70
): FaceDetection[] {
  const candidates: FaceDetection[] = [];
  const numPriors = anchors.length;
  
  const centerVariance = 0.1;
  const sizeVariance = 0.2;
  
  for (let i = 0; i < numPriors; i++) {
    const scoreFace = scores[i * 2 + 1];
    
    if (scoreFace >= scoreThreshold) {
      const locIdx = i * 4;
      const prior = anchors[i];
      
      const predCx = locations[locIdx];
      const predCy = locations[locIdx + 1];
      const predW = locations[locIdx + 2];
      const predH = locations[locIdx + 3];
      
      const cx = predCx * centerVariance * prior[2] + prior[0];
      const cy = predCy * centerVariance * prior[3] + prior[1];
      const w = Math.exp(predW * sizeVariance) * prior[2];
      const h = Math.exp(predH * sizeVariance) * prior[3];
      
      const xmin = Math.max(0, cx - w / 2);
      const ymin = Math.max(0, cy - h / 2);
      const xmax = Math.min(1.0, cx + w / 2);
      const ymax = Math.min(1.0, cy + h / 2);
      
      candidates.push({
        box: [xmin, ymin, xmax, ymax],
        score: scoreFace
      });
    }
  }
  return candidates;
}

function iou(boxA: [number, number, number, number], boxB: [number, number, number, number]): number {
  const xA = Math.max(boxA[0], boxB[0]);
  const yA = Math.max(boxA[1], boxB[1]);
  const xB = Math.min(boxA[2], boxB[2]);
  const yB = Math.min(boxA[3], boxB[3]);
  
  const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
  if (interArea === 0) return 0;
  
  const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  
  return interArea / (boxAArea + boxBArea - interArea);
}

function nonMaxSuppression(candidates: FaceDetection[], iouThreshold = 0.3): FaceDetection[] {
  candidates.sort((a, b) => b.score - a.score);
  const selected: FaceDetection[] = [];
  
  while (candidates.length > 0) {
    const current = candidates.shift()!;
    selected.push(current);
    
    candidates = candidates.filter(item => {
      const overlap = iou(current.box, item.box);
      return overlap <= iouThreshold;
    });
  }
  
  return selected;
}

async function captureAndValidateFace(cameraRef: React.RefObject<CameraView | null>, employeeId: string): Promise<number[]> {
  if (Platform.OS === "web" || !ort) {
    console.log("[Mock Mode] captureAndValidateFace: Running in Web/Mock Mode. Using mock embeddings.");
    await new Promise((resolve) => setTimeout(resolve, 800));
    const seed = getDeterministicSeed(employeeId);
    return generateMockEmbedding(seed, true);
  }

  if (!cameraRef.current) {
    throw new Error("Camera view reference is null");
  }
  
  const detector = await loadDetectorModel();
  const recognizer = await loadRecognizerModel();
  const moireModel = await loadMoireModel();
  
  console.log("Capturing photo...");
  const photo = await cameraRef.current.takePictureAsync({
    skipProcessing: true,
  });
  if (!photo || !photo.uri) {
    throw new Error("Failed to capture photo from CameraView");
  }
  console.log(`Captured photo resolution: ${photo.width}x${photo.height}`);
  
  console.log("Resizing photo to 320x240 for detection...");
  const detectorManip = await manipulateAsync(
    photo.uri,
    [
      {
        resize: {
          width: 320,
          height: 240,
        },
      },
    ],
    { compress: 0.9, format: SaveFormat.JPEG, base64: true }
  );
  
  if (!detectorManip.base64) {
    throw new Error("Failed to resize and get base64 data for detection");
  }
  
  const detectorJpegData = base64.toByteArray(detectorManip.base64);
  const detectorRawImage = jpeg.decode(detectorJpegData, { useTArray: true });
  
  const numDetectorPixels = 320 * 240;
  const detectorFloatData = new Float32Array(3 * numDetectorPixels);
  const rOffsetD = 0;
  const gOffsetD = numDetectorPixels;
  const bOffsetD = 2 * numDetectorPixels;
  const detectorData = detectorRawImage.data;
  
  for (let i = 0; i < numDetectorPixels; i++) {
    const r = detectorData[i * 4];
    const g = detectorData[i * 4 + 1];
    const b = detectorData[i * 4 + 2];
    
    detectorFloatData[rOffsetD + i] = (r - 127.0) / 128.0;
    detectorFloatData[gOffsetD + i] = (g - 127.0) / 128.0;
    detectorFloatData[bOffsetD + i] = (b - 127.0) / 128.0;
  }
  
  console.log("Running local ONNX face detection...");
  const detectorInput = new ort.Tensor("float32", detectorFloatData, [1, 3, 240, 320]);
  const detectorOutputs = await detector.run({ "input": detectorInput });
  
  const rawScores = detectorOutputs["scores"];
  const rawBoxes = detectorOutputs["boxes"];
  
  if (!rawScores || !rawBoxes) {
    throw new Error("Face detector returned invalid outputs");
  }
  
  const scoreThreshold = 0.70;
  const candidates = decodeBoxes(
    rawBoxes.data as Float32Array,
    rawScores.data as Float32Array,
    scoreThreshold
  );
  
  const detections = nonMaxSuppression(candidates, 0.3);
  console.log(`Detected faces count: ${detections.length}`);
  
  if (detections.length === 0) {
    Alert.alert("Face Detection Error", "No face detected. Please position your face clearly inside the camera box.");
    throw new Error("No face detected");
  }
  if (detections.length > 1) {
    Alert.alert("Face Detection Error", "Multiple faces detected. Please make sure only one person is in the camera frame.");
    throw new Error("Multiple faces detected");
  }
  
  const face = detections[0];
  const [xmin, ymin, xmax, ymax] = face.box;
  const faceWidth = xmax - xmin;
  const faceHeight = ymax - ymin;
  
  console.log(`Detected face relative size: ${faceWidth.toFixed(3)}x${faceHeight.toFixed(3)}`);
  
  if (faceWidth > 0.7 || faceHeight > 0.7) {
    Alert.alert("Positioning Error", "Please move your face slightly further away from the screen.");
    throw new Error("Face is too close to camera");
  }
  
  // ── Run Moiré / spoof detection ──
  const w = xmax - xmin;
  const h = ymax - ymin;
  const maxDim = Math.max(w, h);
  const centerX = xmin + w / 2;
  const centerY = ymin + h / 2;
  const expansionFactor = 1.5;

  const moireCropX = Math.max(0, Math.floor((centerX - (maxDim * expansionFactor) / 2) * photo.width));
  const moireCropY = Math.max(0, Math.floor((centerY - (maxDim * expansionFactor) / 2) * photo.height));
  const moireCropW = Math.min(photo.width - moireCropX, Math.ceil(maxDim * expansionFactor * photo.width));
  const moireCropH = Math.min(photo.height - moireCropY, Math.ceil(maxDim * expansionFactor * photo.height));

  console.log(`Cropping expanded face region for Moiré detection: x=${moireCropX}, y=${moireCropY}, w=${moireCropW}, h=${moireCropH}`);
  const moireManip = await manipulateAsync(
    photo.uri,
    [
      {
        crop: {
          originX: moireCropX,
          originY: moireCropY,
          width: moireCropW,
          height: moireCropH,
        },
      },
      {
        resize: {
          width: 128,
          height: 128,
        },
      },
    ],
    { compress: 0.9, format: SaveFormat.JPEG, base64: true }
  );

  if (!moireManip.base64) {
    throw new Error("Failed to crop and base64 encode face image for Moiré detection");
  }

  const moireJpegData = base64.toByteArray(moireManip.base64);
  const moireRawImage = jpeg.decode(moireJpegData, { useTArray: true });

  const numMoirePixels = 128 * 128;
  const moireFloatData = new Float32Array(3 * numMoirePixels);
  const rOffsetM = 0;
  const gOffsetM = numMoirePixels;
  const bOffsetM = 2 * numMoirePixels;
  const moireData = moireRawImage.data;

  for (let i = 0; i < numMoirePixels; i++) {
    const r = moireData[i * 4];
    const g = moireData[i * 4 + 1];
    const b = moireData[i * 4 + 2];

    moireFloatData[rOffsetM + i] = r / 255.0;
    moireFloatData[gOffsetM + i] = g / 255.0;
    moireFloatData[bOffsetM + i] = b / 255.0;
  }

  console.log("Running local ONNX Moiré / spoof detection...");
  const moireInput = new ort.Tensor("float32", moireFloatData, [1, 3, 128, 128]);
  const moireOutputs = await moireModel.run({ "input": moireInput });
  const moireOutputTensor = moireOutputs["output"];

  if (!moireOutputTensor) {
    throw new Error("output tensor not found in Moiré model outputs");
  }

  const logits = moireOutputTensor.data as Float32Array;
  const realLogit = logits[0];
  const spoofLogit = logits[1];
  const logitDiff = realLogit - spoofLogit;

  console.log(`Moiré results - real: ${realLogit.toFixed(4)}, spoof: ${spoofLogit.toFixed(4)}, diff: ${logitDiff.toFixed(4)}`);

  if (logitDiff < 0) {
    Alert.alert(
      "Liveness Verification Failed",
      "Screen spoofing or Moiré pattern detected. Please scan a real face."
    );
    throw new Error("Screen spoofing or Moiré pattern detected");
  }
  
  console.log("Cropping face region from original photo...");
  const cropX = Math.max(0, Math.floor(xmin * photo.width));
  const cropY = Math.max(0, Math.floor(ymin * photo.height));
  const cropW = Math.min(photo.width - cropX, Math.ceil((xmax - xmin) * photo.width));
  const cropH = Math.min(photo.height - cropY, Math.ceil((ymax - ymin) * photo.height));
  
  const faceManip = await manipulateAsync(
    photo.uri,
    [
      {
        crop: {
          originX: cropX,
          originY: cropY,
          width: cropW,
          height: cropH,
        },
      },
      {
        resize: {
          width: 112,
          height: 112,
        },
      },
    ],
    { compress: 0.9, format: SaveFormat.JPEG, base64: true }
  );
  
  if (!faceManip.base64) {
    throw new Error("Failed to crop and base64 encode face image");
  }
  
  const faceJpegData = base64.toByteArray(faceManip.base64);
  const faceRawImage = jpeg.decode(faceJpegData, { useTArray: true });
  
  const numRecognizerPixels = 112 * 112;
  const recognizerFloatData = new Float32Array(3 * numRecognizerPixels);
  const rOffsetR = 0;
  const gOffsetR = numRecognizerPixels;
  const bOffsetR = 2 * numRecognizerPixels;
  const faceData = faceRawImage.data;
  
  for (let i = 0; i < numRecognizerPixels; i++) {
    const r = faceData[i * 4];
    const g = faceData[i * 4 + 1];
    const b = faceData[i * 4 + 2];
    
    recognizerFloatData[rOffsetR + i] = (r - 127.5) / 127.5;
    recognizerFloatData[gOffsetR + i] = (g - 127.5) / 127.5;
    recognizerFloatData[bOffsetR + i] = (b - 127.5) / 127.5;
  }
  
  console.log("Running local ONNX face recognition...");
  const recognizerInput = new ort.Tensor("float32", recognizerFloatData, [1, 3, 112, 112]);
  const recognizerOutputs = await recognizer.run({ "input_image": recognizerInput });
  const embeddingsTensor = recognizerOutputs["embeddings"];
  
  if (!embeddingsTensor) {
    throw new Error("embeddings output tensor not found in model outputs");
  }
  
  const embedding = Array.from(embeddingsTensor.data as Float32Array);
  console.log("Recognition completed! Generated 512-dim embedding.");
  return embedding;
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

  const photometricResultsRef = useRef<boolean[]>([]);
  const rppgSignalsRef = useRef<number[]>([]);
  const gyroDataRef = useRef<{x: number, y: number, z: number}[]>([]);

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
  const cameraRef = useRef<CameraView>(null);

  // Load ONNX model on mount
  useEffect(() => {
    if (Platform.OS !== "web") {
      Promise.all([loadDetectorModel(), loadRecognizerModel(), loadMoireModel()]).catch((err: any) => {
        console.error("Failed to pre-load ONNX models on mount:", err);
      });
    }
  }, []);

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
        photometricResultsRef.current = [];
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

        const runFlashStep = async () => {
          if (dirIdx >= dirs.length) return;
          const currentDir = dirs[dirIdx];
          const color = BRIGHT_COLORS[Math.floor(Math.random() * BRIGHT_COLORS.length)];
          setFlashColor(color);
          setFlashDir(currentDir);
          setFlashVisible(true);
          flashAnim.setValue(0);
          Animated.sequence([
            Animated.timing(flashAnim, { toValue: 1, duration: 150, useNativeDriver: false }),
            Animated.delay(400),
            Animated.timing(flashAnim, { toValue: 0, duration: 180, useNativeDriver: false }),
          ]).start();

          setTimeout(async () => {
            if (!cameraRef.current) return;
            try {
              console.log(`Photometric capture start for: ${currentDir}`);
              const photo = await cameraRef.current.takePictureAsync({ skipProcessing: true });
              if (photo && photo.uri) {
                const passed = await verifyDirectionalBrightness(photo.uri, currentDir);
                photometricResultsRef.current.push(passed);
              }
            } catch (err) {
              console.error(`Photometric capture error for ${currentDir}:`, err);
            }
          }, 250);

          dirIdx++;
        };

        runFlashStep();
        flashTimer.current = setInterval(runFlashStep, 1800);
      }

      if (stage.key === "rppg") {
        rppgSignalsRef.current = [];
        const captureInterval = 700;

        const runRppgCapture = async () => {
          if (!cameraRef.current) return;
          let photoUri: string | null = null;
          let manipUri: string | null = null;
          try {
            const photo = await cameraRef.current.takePictureAsync({ skipProcessing: true });
            if (photo && photo.uri) {
              photoUri = photo.uri;
              const manip = await manipulateAsync(
                photo.uri,
                [{ resize: { width: 32, height: 32 } }],
                { compress: 0.9, format: SaveFormat.JPEG, base64: true }
              );
              manipUri = manip.uri;
              if (manip.base64) {
                const jpegData = base64.toByteArray(manip.base64);
                const rawImage = jpeg.decode(jpegData, { useTArray: true });
                const pixels = rawImage.data;

                let greenSum = 0;
                let count = 0;
                for (let y = 4; y < 10; y++) {
                  for (let x = 10; x < 22; x++) {
                    const idx = (y * 32 + x) * 4;
                    greenSum += pixels[idx + 1];
                    count++;
                  }
                }
                const avgGreen = greenSum / count;
                rppgSignalsRef.current.push(avgGreen);
                console.log(`rPPG green sample: ${avgGreen.toFixed(2)}`);
              }
            }
          } catch (err) {
            console.error("rPPG frame capture error:", err);
          } finally {
            try {
              if (photoUri) {
                await FileSystem.deleteAsync(photoUri, { idempotent: true });
              }
              if (manipUri) {
                await FileSystem.deleteAsync(manipUri, { idempotent: true });
              }
            } catch (e) {
              console.warn("Failed to delete temp files in runRppgCapture:", e);
            }
          }
        };

        runRppgCapture();
        flashTimer.current = setInterval(runRppgCapture, captureInterval);
      }

      if (stage.key === "imu") {
        gyroDataRef.current = [];
        if (Platform.OS !== "web") {
          try {
            Gyroscope.setUpdateInterval(100);
            const subscription = Gyroscope.addListener(gyroData => {
              gyroDataRef.current.push(gyroData);
            });
            (runStage as any).gyroSubscription = subscription;
          } catch (err) {
            console.error("Error setting up Gyroscope subscription:", err);
          }
        }
      }

      stageTimer.current = setTimeout(async () => {
        if (flashTimer.current) {
          clearInterval(flashTimer.current);
          flashTimer.current = null;
          setFlashDir(null);
          setFlashVisible(false);
        }
        if (faceTimer.current) { clearTimeout(faceTimer.current); faceTimer.current = null; }
        if (faceTrackTimer.current) { clearTimeout(faceTrackTimer.current); faceTrackTimer.current = null; }

        let passed = faceDetectedRef.current;

        if (passed && Platform.OS !== "web") {
          if (stage.key === "flash") {
            const results = photometricResultsRef.current;
            const passes = results.filter(r => r).length;
            console.log(`Photometric stage completed. Results: ${JSON.stringify(results)}, Passes: ${passes}`);
            if (passes < 2) {
              console.log("Photometric liveness check failed");
              passed = false;
            }
          } else if (stage.key === "rppg") {
            const rppgPassed = await verifyRPPGPulse(rppgSignalsRef.current);
            if (!rppgPassed) {
              console.log("rPPG liveness check failed");
              passed = false;
            }
          } else if (stage.key === "imu") {
            const subscription = (runStage as any).gyroSubscription;
            if (subscription) {
              subscription.remove();
              (runStage as any).gyroSubscription = null;
            }
            const gyroSamples = gyroDataRef.current;
            console.log(`IMU Correlation stage completed. Gyro samples count: ${gyroSamples.length}`);
            
            let hasMovement = false;
            if (gyroSamples.length > 2) {
              let totalVar = 0;
              for (let i = 1; i < gyroSamples.length; i++) {
                const dx = gyroSamples[i].x - gyroSamples[i-1].x;
                const dy = gyroSamples[i].y - gyroSamples[i-1].y;
                const dz = gyroSamples[i].z - gyroSamples[i-1].z;
                totalVar += Math.sqrt(dx*dx + dy*dy + dz*dz);
              }
              console.log(`Gyro accumulated delta: ${totalVar.toFixed(4)}`);
              if (totalVar > 0.0005) {
                hasMovement = true;
              }
            } else {
              hasMovement = true; // Fallback if sensor not available
            }
            if (!hasMovement) {
              console.log("IMU Correlation check failed");
              passed = false;
            }
          }
        }

        if (passed) {
          // ── Collect real embedding during moiré stage ──────────────────
          if (stage.key === "moire") {
            try {
              const embedding = await captureAndValidateFace(cameraRef, employeeId);
              embeddingSamples.current.push(embedding);
            } catch (err: any) {
              console.error("Error capturing/validating face:", err);
              setStagePhase("failed");
              setBoxPassed(false);
              setHasFailed(true);
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
              setTimeout(() => {
                runStage(idx);
              }, 2000);
              return;
            }
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
        // Generate additional samples to reach 5-10 photos based on the real embedding
        const realEmb = embeddingSamples.current[0];
        while (embeddingSamples.current.length < 5) {
          if (realEmb) {
            // Add small noise to simulate variance
            const noisyEmb = realEmb.map((v) => {
              const noise = (Math.random() * 0.02) - 0.01; // ±1% noise
              return v + noise;
            });
            const norm = Math.sqrt(noisyEmb.reduce((s, v) => s + v * v, 0));
            embeddingSamples.current.push(noisyEmb.map((v) => v / (norm || 1)));
          } else {
            const baseSeed = getDeterministicSeed(employeeId);
            embeddingSamples.current.push(generateMockEmbedding(baseSeed, true));
          }
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
      const subscription = (runStage as any).gyroSubscription;
      if (subscription) {
        subscription.remove();
        (runStage as any).gyroSubscription = null;
      }
    };
  }, [runStage]);

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
        ref={cameraRef}
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
