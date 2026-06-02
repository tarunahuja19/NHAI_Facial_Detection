import { Feather } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { Image } from "expo-image";
import { router } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
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
const DISMISS_AFTER = 8;

export default function SuccessScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const [countdown, setCountdown] = useState(DISMISS_AFTER);

  const circleScale = useRef(new Animated.Value(0)).current;
  const checkOpacity = useRef(new Animated.Value(0)).current;
  const textOpacity = useRef(new Animated.Value(0)).current;
  const ripple1 = useRef(new Animated.Value(0)).current;
  const ripple2 = useRef(new Animated.Value(0)).current;
  const progressAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    Animated.sequence([
      Animated.spring(circleScale, {
        toValue: 1,
        friction: 5,
        tension: 80,
        useNativeDriver: true,
      }),
      Animated.timing(checkOpacity, {
        toValue: 1,
        duration: 300,
        useNativeDriver: true,
      }),
      Animated.timing(textOpacity, {
        toValue: 1,
        duration: 400,
        useNativeDriver: true,
      }),
    ]).start();

    const runRipple = () => {
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
    };
    runRipple();

    Animated.timing(progressAnim, {
      toValue: 0,
      duration: DISMISS_AFTER * 1000,
      useNativeDriver: false,
    }).start();

    const interval = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) {
          clearInterval(interval);
          router.replace("/");
          return 0;
        }
        return c - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const topInset = Platform.OS === "web" ? 67 : insets.top;
  const bottomInset = Platform.OS === "web" ? 34 : insets.bottom;

  const ripple1Scale = ripple1.interpolate({ inputRange: [0, 1], outputRange: [1, 2.4] });
  const ripple1Opacity = ripple1.interpolate({ inputRange: [0, 0.4, 1], outputRange: [0.35, 0.1, 0] });
  const ripple2Scale = ripple2.interpolate({ inputRange: [0, 1], outputRange: [1, 2.0] });
  const ripple2Opacity = ripple2.interpolate({ inputRange: [0, 0.4, 1], outputRange: [0.25, 0.08, 0] });

  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ["0%", "100%"],
  });

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

      <View style={styles.body}>
        <View style={styles.circleWrapper}>
          <Animated.View
            style={[
              styles.ripple,
              {
                backgroundColor: "#22c55e",
                transform: [{ scale: ripple1Scale }],
                opacity: ripple1Opacity,
              },
            ]}
          />
          <Animated.View
            style={[
              styles.ripple,
              {
                backgroundColor: "#22c55e",
                transform: [{ scale: ripple2Scale }],
                opacity: ripple2Opacity,
              },
            ]}
          />
          <Animated.View
            style={[
              styles.successCircle,
              { transform: [{ scale: circleScale }] },
            ]}
          >
            <Animated.View style={{ opacity: checkOpacity }}>
              <Feather name="check" size={52} color="#fff" strokeWidth={3} />
            </Animated.View>
          </Animated.View>
        </View>

        <Animated.View style={[styles.textBlock, { opacity: textOpacity }]}>
          <Text style={[styles.successTitle, { color: "#22c55e" }]}>
            Registered Successfully
          </Text>
          <Text style={[styles.successSub, { color: colors.mutedForeground }]}>
            Identity verified · All 4 liveness checks passed
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
                    { backgroundColor: "#dcfce7" },
                  ]}
                >
                  <Feather name="check" size={12} color="#22c55e" />
                </View>
                <Text style={[styles.checkRowText, { color: colors.foreground }]}>
                  {item.label}
                </Text>
                <Text style={[styles.checkRowStatus, { color: "#22c55e" }]}>
                  Passed
                </Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </View>

      <Animated.View style={[styles.footer, { opacity: textOpacity }]}>
        <View
          style={[
            styles.progressTrack,
            { backgroundColor: colors.border },
          ]}
        >
          <Animated.View
            style={[
              styles.progressFill,
              { width: progressWidth },
            ]}
          />
        </View>
        <Text style={[styles.countdownText, { color: colors.mutedForeground }]}>
          Returning to home in {countdown}s
        </Text>
        <Pressable
          onPress={() => router.replace("/")}
          style={[styles.homeBtn, { backgroundColor: colors.primary }]}
        >
          <Feather name="home" size={18} color="#fff" />
          <Text style={styles.homeBtnText}>Back to Home</Text>
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
  successCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "#22c55e",
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "#22c55e",
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
  successTitle: {
    fontSize: 26,
    fontFamily: "Inter_700Bold",
    textAlign: "center",
  },
  successSub: {
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
  progressTrack: {
    height: 4,
    borderRadius: 2,
    overflow: "hidden",
  },
  progressFill: {
    height: 4,
    backgroundColor: "#22c55e",
    borderRadius: 2,
  },
  countdownText: {
    fontSize: 13,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
  },
  homeBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 16,
    borderRadius: 14,
  },
  homeBtnText: {
    color: "#fff",
    fontSize: 16,
    fontFamily: "Inter_700Bold",
  },
});
