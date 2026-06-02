import { Feather } from "@expo/vector-icons";
import { router } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
import {
  Animated,
  Linking,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useColors } from "@/hooks/useColors";

const REDIRECT_AFTER = 30;

const STEPS = [
  {
    step: "1",
    title: "Open Device Settings",
    desc: 'Tap the gear icon on your phone or search for "Settings"',
  },
  {
    step: "2",
    title: "Find the App",
    desc: 'Go to Apps (or Application Manager) and find "Expo Go"',
  },
  {
    step: "3",
    title: "Open Permissions",
    desc: 'Tap "Permissions" inside the app settings',
  },
  {
    step: "4",
    title: "Enable Camera",
    desc: 'Tap "Camera" and select "Allow" or toggle it on',
  },
];

export default function DeniedScreen() {
  const colors = useColors();
  const insets = useSafeAreaInsets();
  const [countdown, setCountdown] = useState(REDIRECT_AFTER);

  const circleScale = useRef(new Animated.Value(0)).current;
  const iconOpacity = useRef(new Animated.Value(0)).current;
  const contentOpacity = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.spring(circleScale, {
        toValue: 1,
        friction: 5,
        tension: 80,
        useNativeDriver: false,
      }),
      Animated.timing(iconOpacity, {
        toValue: 1,
        duration: 250,
        useNativeDriver: false,
      }),
      Animated.timing(contentOpacity, {
        toValue: 1,
        duration: 350,
        useNativeDriver: false,
      }),
    ]).start();

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

  const openSettings = () => {
    Linking.openSettings();
  };

  return (
    <View
      style={[
        styles.root,
        { backgroundColor: colors.background },
      ]}
    >
      <ScrollView
        contentContainerStyle={[
          styles.scroll,
          { paddingTop: topInset + 16, paddingBottom: bottomInset + 24 },
        ]}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.topSection}>
          <Animated.View
            style={[styles.circle, { transform: [{ scale: circleScale }] }]}
          >
            <Animated.View style={{ opacity: iconOpacity }}>
              <Feather name="camera-off" size={44} color="#fff" />
            </Animated.View>
          </Animated.View>

          <Animated.View style={[styles.headBlock, { opacity: contentOpacity }]}>
            <Text style={[styles.title, { color: "#ef4444" }]}>
              Camera Permission Denied
            </Text>
            <Text style={[styles.subtitle, { color: colors.mutedForeground }]}>
              Camera access is required for liveness detection. Enable it manually by following the steps below.
            </Text>
          </Animated.View>
        </View>

        <Animated.View style={[{ opacity: contentOpacity }]}>
          <Text style={[styles.stepsLabel, { color: colors.foreground }]}>
            How to enable camera access
          </Text>

          {STEPS.map((item, i) => (
            <View
              key={i}
              style={[styles.stepCard, { backgroundColor: colors.card }]}
            >
              <View style={[styles.stepNum, { backgroundColor: "#fee2e2" }]}>
                <Text style={[styles.stepNumText, { color: "#ef4444" }]}>
                  {item.step}
                </Text>
              </View>
              <View style={styles.stepText}>
                <Text style={[styles.stepTitle, { color: colors.foreground }]}>
                  {item.title}
                </Text>
                <Text style={[styles.stepDesc, { color: colors.mutedForeground }]}>
                  {item.desc}
                </Text>
              </View>
            </View>
          ))}

          <Pressable
            onPress={openSettings}
            style={[styles.settingsBtn, { backgroundColor: "#ef4444" }]}
          >
            <Feather name="settings" size={18} color="#fff" />
            <Text style={styles.settingsBtnText}>Open Settings</Text>
          </Pressable>

          <Pressable
            onPress={() => router.replace("/")}
            style={[styles.homeBtn, { borderColor: colors.border }]}
          >
            <Feather name="home" size={16} color={colors.primary} />
            <Text style={[styles.homeBtnText, { color: colors.primary }]}>
              Back to Home ({countdown}s)
            </Text>
          </Pressable>
        </Animated.View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  scroll: {
    paddingHorizontal: 24,
    gap: 0,
  },
  topSection: {
    alignItems: "center",
    marginBottom: 28,
    gap: 20,
  },
  circle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: "#ef4444",
    alignItems: "center",
    justifyContent: "center",
    elevation: 10,
  },
  headBlock: {
    alignItems: "center",
    gap: 8,
  },
  title: {
    fontSize: 22,
    fontFamily: "Inter_700Bold",
    textAlign: "center",
  },
  subtitle: {
    fontSize: 14,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
    lineHeight: 21,
  },
  stepsLabel: {
    fontSize: 15,
    fontFamily: "Inter_700Bold",
    marginBottom: 12,
  },
  stepCard: {
    flexDirection: "row",
    alignItems: "flex-start",
    borderRadius: 12,
    padding: 14,
    marginBottom: 8,
    gap: 12,
  },
  stepNum: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  stepNumText: {
    fontSize: 14,
    fontFamily: "Inter_700Bold",
  },
  stepText: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 14,
    fontFamily: "Inter_600SemiBold",
    marginBottom: 2,
  },
  stepDesc: {
    fontSize: 12,
    fontFamily: "Inter_400Regular",
    lineHeight: 17,
  },
  settingsBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingVertical: 16,
    borderRadius: 14,
    marginTop: 16,
    marginBottom: 10,
  },
  settingsBtnText: {
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
    borderWidth: 1.5,
  },
  homeBtnText: {
    fontSize: 15,
    fontFamily: "Inter_500Medium",
  },
});
