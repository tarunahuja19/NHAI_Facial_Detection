const fs = require("fs");
const path = require("path");

const key = "nhai_secure_model_key_2026_encryption";
const sourceDir = path.resolve(__dirname, "../assets_raw");
const destDir = path.resolve(__dirname, "../assets");

const filesToEncrypt = ["detector.onnx", "face_model_quant.onnx", "moire.onnx"];

console.log("=== NHAI Model Encryption Script ===");

filesToEncrypt.forEach((file) => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(destDir, file);

  if (!fs.existsSync(sourcePath)) {
    console.error(`Error: Source file ${sourcePath} does not exist.`);
    process.exit(1);
  }

  console.log(`Encrypting ${file}...`);
  const data = fs.readFileSync(sourcePath);
  const encrypted = Buffer.alloc(data.length);

  for (let i = 0; i < data.length; i++) {
    const keyChar = key.charCodeAt(i % key.length);
    encrypted[i] = data[i] ^ keyChar;
  }

  fs.writeFileSync(destPath, encrypted);
  console.log(`Successfully encrypted ${file} -> ${destPath} (Size: ${data.length} bytes)`);
});

console.log("=== Encryption Complete ===");
