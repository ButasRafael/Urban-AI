// App.tsx
import React from 'react';
import 'react-native-get-random-values';  
import { useColorScheme } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
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
import { theme as lightTheme, darkTheme } from './src/theme';
import { StatusBar } from 'react-native';
// ← your shared-element navigator + fade-through defaults
import { Stack, screenOptions } from './src/navigation/Stack';

import RegisterScreen    from './src/screens/RegisterScreen';
import LoginScreen       from './src/screens/LoginScreen';
import HomeScreen        from './src/screens/HomeScreen';
import GalleryScreen     from './src/screens/GalleryScreen';
import DetailScreen      from './src/screens/DetailScreen';
import TestRefreshScreen from './src/screens/TestRefreshScreen';

export default function App() {
  const systemScheme = useColorScheme();
  const [themeMode, setThemeMode] = React.useState<'system'|'light'|'dark'>('system');

  // pick actual scheme
  const effectiveScheme = themeMode === 'system' ? systemScheme : themeMode;
  const currentRestyleTheme = effectiveScheme === 'dark' ? darkTheme : lightTheme;
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

  // load fonts
  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
  });
  if (!fontsLoaded) return null;

  // theme toggle handler
  const toggleTheme = () =>
    setThemeMode(m => m === 'light' ? 'dark' : m === 'dark' ? 'system' : 'light');

  // day/night icon helper
  const ThemeIcon = () => (
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
  );

  return (
    <ThemeProvider theme={currentRestyleTheme}>
       <StatusBar
        barStyle={isDark ? 'light-content' : 'dark-content'}
        animated
      />
      <NavigationContainer ref={navigationRef} theme={navTheme}>
        <Stack.Navigator
          initialRouteName="Login"
          screenOptions={({ route }) => {
            // base fade‐through animation
            const opts = {
              ...screenOptions,
              // show toggle on every screen
              headerRight: ThemeIcon,
            };

            if (route.name === 'Login') {
              // no back arrow on Login
              return {
                ...opts,
                headerLeft: () => null,
                title: '',        // or "Sign In"
              };
            }

            return opts;
          }}
        >
          <Stack.Screen name="Login"   component={LoginScreen}   />
          <Stack.Screen name="Register"component={RegisterScreen}/>
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
            name="Detail"
            component={DetailScreen}
            sharedElements={route => [
              `item.${route.params.media.media_id}.photo`,
            ]}
            options={{
              headerBackTitleVisible: false,
              title: '',
              headerTintColor: currentRestyleTheme.colors.text,
            }}
          />
          <Stack.Screen
            name="TestRefresh"
            component={TestRefreshScreen}
            options={{ title: 'Test Refresh' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </ThemeProvider>
  );
}
