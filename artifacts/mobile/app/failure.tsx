import { Feather } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { Image } from "expo-image";
import { router } from "expo-router";
import React, { useEffect, useRef } from "react";
import {
  Animated,
  Dimensions,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useColors } from "@/hooks/useColors";

const { width: SW } = Dimensions.get("window");

export default function FailureScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();

  const circleScale = useRef(new Animated.Value(0)).current;
  const crossOpacity = useRef(new Animated.Value(0)).current;
  const textOpacity = useRef(new Animated.Value(0)).current;
  const ripple1 = useRef(new Animated.Value(0)).current;
  const ripple2 = useRef(new Animated.Value(0)).current;
  const shakeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);

    // Entry animation
    Animated.sequence([
      Animated.spring(circleScale, {
        toValue: 1,
        friction: 5,
        tension: 80,
        useNativeDriver: true,
      }),
      Animated.timing(crossOpacity, {
        toValue: 1,
        duration: 250,
        useNativeDriver: true,
      }),
      Animated.timing(textOpacity, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();

    // Shake the circle once
    setTimeout(() => {
      Animated.sequence([
        Animated.timing(shakeAnim, { toValue: 10, duration: 60, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue: -10, duration: 60, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue: 8, duration: 60, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue: -6, duration: 60, useNativeDriver: true }),
        Animated.timing(shakeAnim, { toValue: 0, duration: 60, useNativeDriver: true }),
      ]).start();
    }, 300);

    // Red ripple loops
    Animated.loop(
      Animated.sequence([
        Animated.timing(ripple1, { toValue: 1, duration: 1400, useNativeDriver: true }),
        Animated.timing(ripple1, { toValue: 0, duration: 0, useNativeDriver: true }),
      ])
    ).start();
    setTimeout(() => {
      Animated.loop(
        Animated.sequence([
          Animated.timing(ripple2, { toValue: 1, duration: 1400, useNativeDriver: true }),
          Animated.timing(ripple2, { toValue: 0, duration: 0, useNativeDriver: true }),
        ])
      ).start();
    }, 700);
  }, []);

  const topInset = Platform.OS === "web" ? 67 : insets.top;
  const bottomInset = Platform.OS === "web" ? 34 : insets.bottom;

  const ripple1Scale = ripple1.interpolate({ inputRange: [0, 1], outputRange: [1, 2.4] });
  const ripple1Opacity = ripple1.interpolate({ inputRange: [0, 0.4, 1], outputRange: [0.3, 0.08, 0] });
  const ripple2Scale = ripple2.interpolate({ inputRange: [0, 1], outputRange: [1, 2.0] });
  const ripple2Opacity = ripple2.interpolate({ inputRange: [0, 0.4, 1], outputRange: [0.2, 0.06, 0] });

  return (
    <View
      style={[
        styles.root,
        {
          backgroundColor: colors.background,
          paddingTop: topInset,
          paddingBottom: bottomInset,
        },
      ]}
    >
      {/* Top bar */}
      <View style={styles.topBar}>
        <Image
          source={require("@/assets/images/nhai_logo.jpg")}
          style={styles.logoSmall}
          contentFit="contain"
        />
        <Text style={[styles.appNameSmall, { color: colors.primary }]}>
          NHAI Liveness Detection
        </Text>
      </View>

      {/* Body */}
      <View style={styles.body}>
        {/* Animated red X circle with ripples */}
        <View style={styles.circleWrapper}>
          <Animated.View
            style={[
              styles.ripple,
              {
                backgroundColor: "#ef4444",
                transform: [{ scale: ripple1Scale }],
                opacity: ripple1Opacity,
              },
            ]}
          />
          <Animated.View
            style={[
              styles.ripple,
              {
                backgroundColor: "#ef4444",
                transform: [{ scale: ripple2Scale }],
                opacity: ripple2Opacity,
              },
            ]}
          />
          <Animated.View
            style={[
              styles.failCircle,
              {
                transform: [
                  { scale: circleScale },
                  { translateX: shakeAnim },
                ],
              },
            ]}
          >
            <Animated.View style={{ opacity: crossOpacity }}>
              <Feather name="x" size={52} color="#fff" strokeWidth={3} />
            </Animated.View>
          </Animated.View>
        </View>

        {/* Text + card */}
        <Animated.View style={[styles.textBlock, { opacity: textOpacity }]}>
          <Text style={[styles.failTitle, { color: "#ef4444" }]}>
            Verification Failed
          </Text>
          <Text style={[styles.failSub, { color: colors.mutedForeground }]}>
            Liveness check could not be completed
          </Text>

          <View style={[styles.detailsCard, { backgroundColor: colors.card }]}>
            {[
              { label: "Screen Flash 3D", icon: "weather-lightning" },
              { label: "Moiré Detection", icon: "grid" },
              { label: "rPPG Pulse", icon: "heart-pulse" },
              { label: "IMU Correlation", icon: "axis-arrow" },
            ].map((item, i) => (
              <View
                key={i}
                style={[
                  styles.checkRow,
                  i < 3 && {
                    borderBottomWidth: 1,
                    borderBottomColor: colors.border,
                  },
                ]}
              >
                <View
                  style={[
                    styles.checkRowDot,
                    { backgroundColor: "rgba(239,68,68,0.12)" },
                  ]}
                >
                  <Feather name="x" size={12} color="#ef4444" />
                </View>
                <Text style={[styles.checkRowText, { color: colors.foreground }]}>
                  {item.label}
                </Text>
                <Text style={[styles.checkRowStatus, { color: "#ef4444" }]}>
                  Failed
                </Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </View>

      {/* Footer */}
      <Animated.View style={[styles.footer, { opacity: textOpacity }]}>
        <Pressable
          onPress={() => router.replace("/liveness")}
          style={styles.retryBtn}
        >
          <Feather name="refresh-cw" size={18} color="#fff" />
          <Text style={styles.retryBtnText}>Restart Verification</Text>
        </Pressable>

        <Pressable
          onPress={() => router.replace("/")}
          style={[styles.homeBtn, { borderColor: colors.border }]}
        >
          <Feather name="home" size={16} color={colors.mutedForeground} />
          <Text style={[styles.homeBtnText, { color: colors.mutedForeground }]}>
            Back to Home
          </Text>
        </Pressable>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    paddingHorizontal: 24,
  },
  topBar: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingTop: 8,
    paddingBottom: 8,
  },
  logoSmall: {
    width: 36,
    height: 36,
    borderRadius: 18,
  },
  appNameSmall: {
    fontSize: 14,
    fontFamily: "Inter_700Bold",
  },
  body: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 32,
  },
  circleWrapper: {
    alignItems: "center",
    justifyContent: "center",
    width: 140,
    height: 140,
  },
  ripple: {
    position: "absolute",
    width: 120,
    height: 120,
    borderRadius: 60,
  },
  failCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "#ef4444",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#ef4444",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.4,
    shadowRadius: 20,
    elevation: 12,
  },
  textBlock: {
    alignItems: "center",
    width: "100%",
    gap: 8,
  },
  failTitle: {
    fontSize: 26,
    fontFamily: "Inter_700Bold",
    textAlign: "center",
  },
  failSub: {
    fontSize: 14,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
    marginBottom: 20,
  },
  detailsCard: {
    width: "100%",
    borderRadius: 16,
    overflow: "hidden",
  },
  checkRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 13,
    gap: 12,
  },
  checkRowDot: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  checkRowText: {
    flex: 1,
    fontSize: 14,
    fontFamily: "Inter_500Medium",
  },
  checkRowStatus: {
    fontSize: 13,
    fontFamily: "Inter_600SemiBold",
  },
  footer: {
    gap: 12,
    paddingBottom: 8,
  },
  retryBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 16,
    borderRadius: 14,
    backgroundColor: "#ef4444",
    shadowColor: "#ef4444",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 6,
  },
  retryBtnText: {
    color: "#fff",
    fontSize: 16,
    fontFamily: "Inter_700Bold",
  },
  homeBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 14,
    borderRadius: 14,
    borderWidth: 1,
  },
  homeBtnText: {
    fontSize: 15,
    fontFamily: "Inter_500Medium",
  },
});
