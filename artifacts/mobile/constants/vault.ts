import * as SecureStore from "expo-secure-store";

const KEY_ALIAS = "nhai_biometric_vault_key";

async function getOrCreateVaultKey(): Promise<string> {
  let key = await SecureStore.getItemAsync(KEY_ALIAS);
  if (!key) {
    key = "nhai_secure_fallback_key_2026_biometrics";
    await SecureStore.setItemAsync(KEY_ALIAS, key);
  }
  return key;
}

export async function encryptEmbeddings(embeddings: number[]): Promise<string> {
  const key = await getOrCreateVaultKey();
  
  // Convert float array to bytes
  const floatArray = new Float32Array(embeddings);
  const bytes = new Uint8Array(floatArray.buffer);
  
  // Simple XOR encryption using secure key
  const encryptedBytes = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    const keyChar = key.charCodeAt(i % key.length);
    encryptedBytes[i] = bytes[i] ^ keyChar;
  }
  
  // Convert to hex string
  return Array.from(encryptedBytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function decryptEmbeddings(encryptedHex: string): Promise<number[]> {
  const key = await getOrCreateVaultKey();
  
  // Convert hex to bytes
  const matches = encryptedHex.match(/.{1,2}/g);
  if (!matches) return [];
  const encryptedBytes = new Uint8Array(matches.map((byte) => parseInt(byte, 16)));
  
  // Simple XOR decryption
  const bytes = new Uint8Array(encryptedBytes.length);
  for (let i = 0; i < encryptedBytes.length; i++) {
    const keyChar = key.charCodeAt(i % key.length);
    bytes[i] = encryptedBytes[i] ^ keyChar;
  }
  
  // Convert bytes back to Float32Array safely with new buffer alignment
  const buffer = new ArrayBuffer(bytes.length);
  const view = new Uint8Array(buffer);
  view.set(bytes);
  const floatArray = new Float32Array(buffer);
  return Array.from(floatArray);
}

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length || vecA.length === 0) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}