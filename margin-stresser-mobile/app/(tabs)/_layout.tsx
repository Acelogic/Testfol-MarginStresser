import { Tabs as ExpoTabs, withLayoutContext } from 'expo-router';
// @ts-ignore - unstable import, types might not be perfectly resolved without specific config
import { createNativeBottomTabNavigator } from '@react-navigation/bottom-tabs/unstable';
import React from 'react';
import { Platform } from 'react-native';
import { IconSymbol } from '@/components/ui/icon-symbol';
import { Colors } from '@/constants/theme';
import { useColorScheme } from '@/hooks/use-color-scheme';
import { HapticTab } from '@/components/haptic-tab';
import { ParamListBase, TabNavigationState } from '@react-navigation/native';
import { NativeBottomTabNavigationOptions, NativeBottomTabNavigationEventMap } from '@react-navigation/bottom-tabs/unstable';

// Create the Native Navigator for iOS
let NativeTabs: any;
if (Platform.OS === 'ios') {
  const { Navigator } = createNativeBottomTabNavigator();
  NativeTabs = withLayoutContext<
    NativeBottomTabNavigationOptions,
    typeof Navigator,
    TabNavigationState<ParamListBase>,
    NativeBottomTabNavigationEventMap
  >(Navigator);
}

export default function TabLayout() {
  const colorScheme = useColorScheme();
  const isNative = Platform.OS === 'ios';

  if (isNative) {
    // Native iOS Layout (Liquid Glass)
    return (
      <NativeTabs
        screenOptions={{
          tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
          headerShown: false,
        }}>
        <NativeTabs.Screen
          name="index"
          options={{
            title: 'Home',
            tabBarLabel: 'Home',
            // @ts-ignore
            tabBarSystemItem: 'favorites',
          }}
        />
        <NativeTabs.Screen
          name="scenarios"
          options={{
            title: 'Saved',
            tabBarLabel: 'Saved',
            // @ts-ignore
            tabBarSystemItem: 'bookmarks',
          }}
        />
        <NativeTabs.Screen
          name="history"
          options={{
            title: 'History',
            tabBarLabel: 'History',
            // @ts-ignore
            tabBarSystemItem: 'recents',
          }}
        />
        <NativeTabs.Screen
          name="guide"
          options={{
            title: 'Guide',
            tabBarLabel: 'Guide',
            // @ts-ignore
            tabBarSystemItem: 'downloads',
          }}
        />
        <NativeTabs.Screen
          name="settings"
          options={{
            title: 'Settings',
            tabBarLabel: 'Settings',
            // @ts-ignore
            tabBarSystemItem: 'more',
          }}
        />
      </NativeTabs>
    );
  }

  // Web / Android Layout (Standard)
  return (
    <ExpoTabs
      screenOptions={{
        tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: Platform.select({
          default: {},
        }),
      }}>
      <ExpoTabs.Screen
        name="index"
        options={{
          title: 'Home',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="house.fill" color={color} />,
        }}
      />
      <ExpoTabs.Screen
        name="scenarios"
        options={{
          title: 'Saved',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="star.fill" color={color} />,
        }}
      />
      <ExpoTabs.Screen
        name="history"
        options={{
          title: 'History',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="clock.fill" color={color} />,
        }}
      />
      <ExpoTabs.Screen
        name="guide"
        options={{
          title: 'Guide',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="book.fill" color={color} />,
        }}
      />
      <ExpoTabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color }) => <IconSymbol size={28} name="gear" color={color} />,
        }}
      />
    </ExpoTabs>
  );
}
