import React from 'react';
import 'react-native-get-random-values';
import { useColorScheme, Pressable, View, Text } from 'react-native';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';
import * as Linking from 'expo-linking';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  NavigationContainer,
  DefaultTheme as LightNavTheme,
  DarkTheme   as DarkNavTheme,
} from '@react-navigation/native';
import { ThemeProvider } from '@shopify/restyle';
import {
  useFonts,
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
} from '@expo-google-fonts/inter';

import { navigationRef }         from './src/navigation/RootNavigation';
import { RootStackParamList }    from './src/navigation/types';
import { lightTheme, darkTheme } from './src/theme';
import { StatusBar } from 'expo-status-bar';
import { Stack, screenOptions } from './src/navigation/Stack';

import RegisterScreen      from './src/screens/RegisterScreen';
import LoginScreen         from './src/screens/LoginScreen';
import VerifyEmailScreen   from './src/screens/VerifyEmailScreen';
import ForgotPasswordScreen from './src/screens/ForgotPasswordScreen';
import ResetPasswordScreen from './src/screens/ResetPasswordScreen';
import HomeScreen          from './src/screens/HomeScreen';
import GalleryScreen       from './src/screens/GalleryScreen';
import ProcessingScreen    from './src/screens/ProcessingScreen';
import DetailScreen        from './src/screens/DetailScreen';
import NotificationSettingsScreen from './src/screens/NotificationSettingsScreen';
import NotificationsScreen from './src/screens/NotificationsScreen';
import { RootToaster } from './src/components/ui/Toast';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { NotificationProvider, useNotifications } from './src/contexts/NotificationContext';

function AppContent({ isAuthed }: { isAuthed: boolean }) {
  const systemScheme = useColorScheme();
  const [themeMode, setThemeMode] = React.useState<'system'|'light'|'dark'>('system');
  const { unreadCount } = useNotifications();

  const effectiveScheme = themeMode === 'system' ? systemScheme : themeMode;
  const currentRestyleTheme = effectiveScheme === 'dark' ? darkTheme : lightTheme
  const baseNavTheme        = effectiveScheme === 'dark' ? DarkNavTheme : LightNavTheme;
  const isDark = effectiveScheme === 'dark';

  const navTheme = React.useMemo(() => ({
    ...baseNavTheme,
    colors: {
      ...baseNavTheme.colors,
      background: currentRestyleTheme.colors.background,
      card:       currentRestyleTheme.colors.card,
      text:       currentRestyleTheme.colors.text,
      primary:    currentRestyleTheme.colors.primary500,
    },
  }), [baseNavTheme, currentRestyleTheme]);

  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
  });

  const toggleTheme = React.useCallback(() => {
    setThemeMode(m => m === 'light' ? 'dark' : m === 'dark' ? 'system' : 'light');
  }, []);

  const NotificationBell = React.useCallback(() => (
    <Pressable
      onPress={() => {
        navigationRef.current?.navigate('Notifications');
      }}
      hitSlop={10}
      style={{ marginRight: 12, position: 'relative' }}
    >
      <Feather name="bell" size={22} color={currentRestyleTheme.colors.primary500} />
      {unreadCount > 0 && (
        <View
          style={{
            position: 'absolute',
            top: -4,
            right: -6,
            backgroundColor: currentRestyleTheme.colors.error,
            borderRadius: 10,
            minWidth: 18,
            height: 18,
            alignItems: 'center',
            justifyContent: 'center',
            paddingHorizontal: 4,
          }}
        >
          <Text style={{ fontSize: 10, color: 'white', fontWeight: 'bold' }}>
            {unreadCount > 99 ? '99+' : unreadCount}
          </Text>
        </View>
      )}
    </Pressable>
  ), [unreadCount, currentRestyleTheme.colors]);

  const handleDeepLink = React.useCallback((url: string): void => {
    const parsed = Linking.parse(url);
    console.log('Deep link received:', parsed);

    // Wait for navigation to be ready before navigating
    // Max 50 retries (5 seconds total) to prevent infinite loop
    let retryCount = 0;
    const MAX_RETRIES = 50;

    const navigate = () => {
      if (!navigationRef.current) {
        retryCount++;
        if (retryCount >= MAX_RETRIES) {
          console.error('Navigation ref not ready after maximum retries');
          return;
        }
        // Navigation not ready yet, retry after a short delay
        setTimeout(navigate, 100);
        return;
      }

      // Handle verify-email deep link
      if (parsed.path === 'verify-email' && parsed.queryParams?.token) {
        navigationRef.current.navigate('VerifyEmail', {
          token: parsed.queryParams.token as string
        });
      }

      // Handle reset-password deep link
      if (parsed.path === 'reset-password' && parsed.queryParams?.token) {
        navigationRef.current.navigate('ResetPassword', {
          token: parsed.queryParams.token as string
        });
      }
    };

    navigate();
  }, []);

  // Handle deep links
  React.useEffect(() => {
    // Handle deep link when app is opened from closed state
    Linking.getInitialURL().then((url: string | null) => {
      if (url) {
        handleDeepLink(url);
      }
    });

    // Handle deep link when app is already open
    const subscription = Linking.addEventListener('url', ({ url }: { url: string }) => {
      handleDeepLink(url);
    });

    return () => subscription.remove();
  }, [handleDeepLink]);

  const ThemeIcon = React.useCallback(() => (
    <MaterialCommunityIcons
      name={
        themeMode === 'dark'    ? 'weather-night'
      : themeMode === 'light'   ? 'white-balance-sunny'
                                : 'theme-light-dark'
      }
      size={22}
      color={currentRestyleTheme.colors.primary500}
      onPress={toggleTheme}
      style={{ marginRight: 12 }}
    />
  ), [themeMode, currentRestyleTheme.colors.primary500, toggleTheme]);

  if (!fontsLoaded) return null;

  return (
    <SafeAreaProvider>
    <ThemeProvider theme={currentRestyleTheme}>
       <StatusBar
        style={isDark ? 'light' : 'dark'}
        animated
      />
      <NavigationContainer ref={navigationRef} theme={navTheme}>
        <Stack.Navigator
          initialRouteName="Login"
          screenOptions={({ route }: { route: { name: string } }) => {
            // Auth screens where bell should not appear
            const authScreens = ['Login', 'Register', 'VerifyEmail', 'ForgotPassword', 'ResetPassword'];
            const isAuthScreen = authScreens.includes(route.name);

            const opts = {
              ...screenOptions,
              headerRight: () => (
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  {isAuthed && !isAuthScreen && <NotificationBell />}
                  <ThemeIcon />
                </View>
              ),
            };

            if (route.name === 'Login') {
              return {
                ...opts,
                headerLeft: () => null,
                title: '',
              };
            }

            return opts;
          }}
        >
          <Stack.Screen name="Login"   component={LoginScreen}   />
          <Stack.Screen name="Register" component={RegisterScreen}/>
          <Stack.Screen
            name="VerifyEmail"
            component={VerifyEmailScreen}
            options={{ title: 'Verificare Email' }}
          />
          <Stack.Screen
            name="ForgotPassword"
            component={ForgotPasswordScreen}
            options={{ title: 'Resetare Parolă' }}
          />
          <Stack.Screen
            name="ResetPassword"
            component={ResetPasswordScreen}
            options={{ title: 'Parolă Nouă' }}
          />
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ title: 'Upload' }}
          />
          <Stack.Screen
            name="Gallery"
            component={GalleryScreen}
            options={{ title: 'My Uploads' }}
          />
          <Stack.Screen
            name="Processing"
            component={ProcessingScreen}
            options={{ title: 'Processing Upload' }}
          />
          <Stack.Screen
            name="Detail"
            component={DetailScreen}
            options={{
              title: '',
              headerTintColor: currentRestyleTheme.colors.text,
            }}
          />
          <Stack.Screen
            name="NotificationSettings"
            component={NotificationSettingsScreen}
            options={{ title: 'Setări Notificări' }}
          />
          <Stack.Screen
            name="Notifications"
            component={NotificationsScreen}
            options={{ title: 'Notificări' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
      <RootToaster />
    </ThemeProvider>
    </SafeAreaProvider>
  );
}

export default function App() {
  const [isAuthed, setIsAuthed] = React.useState(false);

  // Check auth status on mount
  React.useEffect(() => {
    async function checkAuth() {
      const token = await AsyncStorage.getItem('accessToken');

      if (token) {
        try {
          const parts = token.split('.');
          if (parts.length === 3) {
            let base64 = parts[1].replace(/-/g, '+').replace(/_/g, '/');
            while (base64.length % 4) {
              base64 += '=';
            }
            const payload = JSON.parse(atob(base64));
            const expiresAt = payload.exp * 1000;
            const now = Date.now();

            if (expiresAt > now) {
              setIsAuthed(true);
              return;
            }
          }
        } catch (error) {
          console.log('Invalid token format, clearing tokens');
        }

        await AsyncStorage.multiRemove(['accessToken', 'refreshToken']);
      }

      setIsAuthed(false);
    }
    checkAuth();

    const unsubscribe = navigationRef.addListener('state', () => {
      checkAuth();
    });

    return unsubscribe;
  }, []);

  return (
    <NotificationProvider enabled={isAuthed}>
      <AppContent isAuthed={isAuthed} />
    </NotificationProvider>
  );
}
