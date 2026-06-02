import * as Haptics from "expo-haptics";
import { Image } from "expo-image";
import { router } from "expo-router";
import React, { useEffect, useRef } from "react";
import {
  Animated,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useColors } from "@/hooks/useColors";

const FEATURES = [
  {
    title: "Screen Flash 3D",
    desc: "Photometric stereo reconstruction using directional screen flashes to detect real 3D facial depth",
  },
  {
    title: "Moiré Detection",
    desc: "Fast Fourier Transform texture analysis to detect screen pixel patterns, blocks photo/video spoofing",
  },
  {
    title: "rPPG Pulse",
    desc: "Camera-based heart rate detection via micro colour changes in forehead skin caused by blood flow",
  },
  {
    title: "IMU Correlation",
    desc: "Gyroscope + optical flow cross-correlation, real faces move in sync with device motion",
  },
];

export default function HomeScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const logoAnim = useRef(new Animated.Value(0)).current;
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const btnScale = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(logoAnim, {
        toValue: 1,
        duration: 700,
        useNativeDriver: false,
      }),
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 700,
        useNativeDriver: false,
      }),
    ]).start();
  }, []);

  const handleStart = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Animated.sequence([
      Animated.timing(btnScale, {
        toValue: 0.96,
        duration: 80,
        useNativeDriver: false,
      }),
      Animated.timing(btnScale, {
        toValue: 1,
        duration: 80,
        useNativeDriver: false,
      }),
    ]).start(() => {
      router.push("/liveness");
    });
  };

  const topInset = Platform.OS === "web" ? 67 : insets.top;
  const bottomInset = Platform.OS === "web" ? 34 : insets.bottom;

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <ScrollView
        contentContainerStyle={[
          styles.scroll,
          { paddingTop: topInset + 16, paddingBottom: bottomInset + 100 },
        ]}
        showsVerticalScrollIndicator={false}
      >
        <Animated.View
          style={[
            styles.header,
            {
              opacity: logoAnim,
              transform: [
                {
                  translateY: logoAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [-24, 0],
                  }),
                },
              ],
            },
          ]}
        >
          <View style={styles.logoContainer}>
            <Image
              source={require("@/assets/images/nhai_logo.jpg")}
              style={styles.logo}
              contentFit="contain"
            />
          </View>
          <Text style={[styles.appName, { color: colors.primary }]}>
            NHAI Liveness Detection
          </Text>
        </Animated.View>

        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={[styles.sectionTitle, { color: colors.foreground }]}>
            Features
          </Text>

          {FEATURES.map((f, i) => (
            <View
              key={i}
              style={[styles.featureCard, { backgroundColor: colors.card }]}
            >
              <View
                style={[
                  styles.featureNum,
                  { backgroundColor: colors.primary + "15" },
                ]}
              >
                <Text style={[styles.featureNumText, { color: colors.primary }]}>
                  {i + 1}
                </Text>
              </View>
              <View style={styles.featureText}>
                <Text style={[styles.featureName, { color: colors.foreground }]}>
                  {f.title}
                </Text>
                <Text
                  style={[styles.featureDesc, { color: colors.mutedForeground }]}
                >
                  {f.desc}
                </Text>
              </View>
            </View>
          ))}
        </Animated.View>
      </ScrollView>

      <Animated.View
        style={[
          styles.btnWrapper,
          {
            bottom: bottomInset + 24,
            transform: [{ scale: btnScale }],
            opacity: fadeAnim,
          },
        ]}
      >
        <Pressable onPress={handleStart} style={styles.btnPressable}>
          <View style={styles.startBtn}>
            <Text style={styles.startBtnText}>Start Liveness Detection</Text>
          </View>
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  scroll: {
    paddingHorizontal: 20,
  },
  header: {
    alignItems: "center",
    marginBottom: 28,
  },
  logoContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "#fff",
    elevation: 6,
    overflow: "hidden",
    marginBottom: 16,
  },
  logo: {
    width: 120,
    height: 120,
  },
  appName: {
    fontSize: 22,
    fontFamily: "Inter_700Bold",
    textAlign: "center",
  },
  sectionTitle: {
    fontSize: 16,
    fontFamily: "Inter_700Bold",
    marginBottom: 12,
  },
  featureCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
    gap: 12,
  },
  featureNum: {
    width: 30,
    height: 30,
    borderRadius: 15,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    marginTop: 1,
  },
  featureNumText: {
    fontSize: 13,
    fontFamily: "Inter_700Bold",
  },
  featureText: {
    flex: 1,
  },
  featureName: {
    fontSize: 14,
    fontFamily: "Inter_600SemiBold",
    marginBottom: 3,
  },
  featureDesc: {
    fontSize: 12,
    fontFamily: "Inter_400Regular",
    lineHeight: 17,
  },
  btnWrapper: {
    position: "absolute",
    left: 20,
    right: 20,
  },
  btnPressable: {
    borderRadius: 16,
    overflow: "hidden",
    elevation: 10,
  },
  startBtn: {
    backgroundColor: "#003087",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 18,
    paddingHorizontal: 24,
  },
  startBtnText: {
    color: "#fff",
    fontSize: 17,
    fontFamily: "Inter_700Bold",
  },
});
