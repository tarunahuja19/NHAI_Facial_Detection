const { getDefaultConfig } = require("expo/metro-config");
const fs = require("fs");

const config = getDefaultConfig(__dirname);

config.watchFolders = (config.watchFolders || []).filter((folder) =>
  fs.existsSync(folder),
);

if (config.resolver && Array.isArray(config.resolver.nodeModulesPaths)) {
  config.resolver.nodeModulesPaths =
    config.resolver.nodeModulesPaths.filter((folder) => fs.existsSync(folder));
}

config.resolver.blockList = [
  /[/\\]android[/\\]/,
  ...((config.resolver && config.resolver.blockList) || []),
];

config.resolver.assetExts.push("onnx");

module.exports = config;
