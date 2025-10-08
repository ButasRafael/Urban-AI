import React, { useEffect, useState } from 'react';
import { Switch, ActivityIndicator, ScrollView, Pressable, Platform } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useTheme } from '@shopify/restyle';
import { Feather } from '@expo/vector-icons';
import DateTimePicker from '@react-native-community/datetimepicker';

import { RootStackParamList } from '../navigation/types';
import { Box, Text } from '../components/restylePrimitives';
import Screen from '../components/ui/Screen';
import HeroHeader from '../components/ui/HeroHeader';
import Card from '../components/ui/Card';
import Badge from '../components/ui/Badge';
import { notify } from '../components/ui/Toast';
import type { Theme } from '../theme';
import {
  getNotificationPreferences,
  updateNotificationPreferences,
  type NotificationPreferences,
} from '../api/notifications';

type Props = NativeStackScreenProps<RootStackParamList, 'NotificationSettings'>;

export default function NotificationSettingsScreen({ navigation }: Props) {
  const theme = useTheme<Theme>();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [preferences, setPreferences] = useState<NotificationPreferences | null>(null);
  const [showStartPicker, setShowStartPicker] = useState(false);
  const [showEndPicker, setShowEndPicker] = useState(false);

  useEffect(() => {
    loadPreferences();
  }, []);

  async function loadPreferences() {
    try {
      const prefs = await getNotificationPreferences();
      setPreferences(prefs);
    } catch (error: any) {
      notify.error('Eroare', 'Nu s-au putut încărca preferințele');
    } finally {
      setLoading(false);
    }
  }

  async function updatePreference(updates: Partial<NotificationPreferences>) {
    if (!preferences) return;

    setSaving(true);
    try {
      const updated = await updateNotificationPreferences(updates);
      setPreferences(updated);
      notify.success('Salvat', 'Preferințele au fost actualizate');
    } catch (error) {
      notify.error('Eroare', 'Nu s-au putut salva preferințele');
    } finally {
      setSaving(false);
    }
  }

  async function handleEventPreferenceChange(
    event: string,
    channel: 'email' | 'push',
    enabled: boolean
  ) {
    if (!preferences) return;

    const newEventPrefs = {
      ...preferences.event_preferences,
      [event]: {
        ...preferences.event_preferences[event],
        [channel]: enabled,
      },
    };

    await updatePreference({ event_preferences: newEventPrefs });
  }

  function formatEventName(eventType: string): string {
    const mapping: Record<string, string> = {
      'media_created': 'Media creată',
      'issue_verified': 'Problemă verificată',
      'issue_assigned': 'Problemă asignată',
      'issue_status_changed': 'Status schimbat',
      'issue_severity_changed': 'Severitate schimbată',
    };
    return mapping[eventType] || eventType.replace(/_/g, ' ');
  }

  function parseTimeString(timeStr: string): Date {
    const [hours, minutes] = timeStr.split(':').map(Number);
    const date = new Date();
    date.setHours(hours);
    date.setMinutes(minutes);
    date.setSeconds(0);
    date.setMilliseconds(0);
    return date;
  }

  function formatTimeString(date: Date): string {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  }

  function handleStartTimeChange(event: any, selectedDate?: Date) {
    setShowStartPicker(Platform.OS === 'ios');
    if (selectedDate && preferences) {
      const timeString = formatTimeString(selectedDate);
      updatePreference({
        quiet_hours: {
          ...preferences.quiet_hours,
          start: timeString,
        },
      });
    }
  }

  function handleEndTimeChange(event: any, selectedDate?: Date) {
    setShowEndPicker(Platform.OS === 'ios');
    if (selectedDate && preferences) {
      const timeString = formatTimeString(selectedDate);
      updatePreference({
        quiet_hours: {
          ...preferences.quiet_hours,
          end: timeString,
        },
      });
    }
  }

  if (loading) {
    return (
      <Screen edges={['left', 'right', 'bottom']}>
        <Box flex={1} alignItems="center" justifyContent="center">
          <ActivityIndicator size="large" color={theme.colors.primary500} />
        </Box>
      </Screen>
    );
  }

  if (!preferences) {
    return (
      <Screen edges={['left', 'right', 'bottom']}>
        <Box flex={1} alignItems="center" justifyContent="center" padding="l">
          <Text variant="body" color="text" textAlign="center">
            Nu s-au putut încărca preferințele
          </Text>
        </Box>
      </Screen>
    );
  }

  return (
    <Screen
      scroll
      padded={false}
      edges={['left', 'right']}
      center
      maxWidth={640}
      dismissKeyboardOnTap
    >
      <HeroHeader
        title="Notificări"
        subtitle="Gestionează cum primești notificările"
        height={180}
        blobs={true}
        rightSlot={
          <Box
            width={56}
            height={56}
            borderRadius="lg"
            style={{
              backgroundColor: 'rgba(255,255,255,0.15)',
              borderWidth: 1,
              borderColor: 'rgba(255,255,255,0.25)',
              ...theme.shadows.sm,
            }}
            alignItems="center"
            justifyContent="center"
          >
            <Feather name="bell" size={24} color="#fff" />
          </Box>
        }
      />

      <Box paddingHorizontal="l" style={{ marginTop: -32 }}>
        {/* Notification Channels */}
        <Box marginBottom="l">
        <Card variant="elevated" padding="l">
          <Box flexDirection="row" alignItems="center" marginBottom="m">
            <Box
              width={28}
              height={28}
              borderRadius="m"
              backgroundColor="primary500"
              marginRight="s"
              alignItems="center"
              justifyContent="center"
              style={theme.shadows.sm}
            >
              <Feather name="settings" size={14} color="#fff" />
            </Box>
            <Text variant="subtitle" color="text">
              Canale de notificare
            </Text>
          </Box>

          {/* Email Toggle */}
          <Box
            flexDirection="row"
            alignItems="center"
            justifyContent="space-between"
            paddingVertical="m"
            borderBottomWidth={1}
            borderBottomColor="border"
          >
            <Box flexDirection="row" alignItems="center" flex={1}>
              <Box
                width={40}
                height={40}
                borderRadius="m"
                backgroundColor="primary100"
                alignItems="center"
                justifyContent="center"
                marginRight="m"
              >
                <Feather name="mail" size={18} color={theme.colors.primary700} />
              </Box>
              <Box flex={1}>
                <Text variant="label" color="text" marginBottom="xs">
                  Email
                </Text>
                <Text variant="caption" color="muted">
                  Primește notificări pe email
                </Text>
              </Box>
            </Box>
            <Switch
              value={preferences.email_enabled}
              onValueChange={(value) => updatePreference({ email_enabled: value })}
              disabled={saving}
              trackColor={{ false: theme.colors.muted, true: theme.colors.primary500 }}
              thumbColor={preferences.email_enabled ? theme.colors.primary700 : theme.colors.surface200}
            />
          </Box>

          {/* Push Notifications Toggle */}
          <Box
            flexDirection="row"
            alignItems="center"
            justifyContent="space-between"
            paddingTop="m"
          >
            <Box flexDirection="row" alignItems="center" flex={1}>
              <Box
                width={40}
                height={40}
                borderRadius="m"
                backgroundColor="info"
                style={{ backgroundColor: 'rgba(52, 152, 219, 0.15)' }}
                alignItems="center"
                justifyContent="center"
                marginRight="m"
              >
                <Feather name="smartphone" size={18} color={theme.colors.info} />
              </Box>
              <Box flex={1}>
                <Text variant="label" color="text" marginBottom="xs">
                  Push Notifications
                </Text>
                <Text variant="caption" color="muted">
                  Notificări în aplicație
                </Text>
              </Box>
            </Box>
            <Switch
              value={preferences.push_enabled}
              onValueChange={(value) => updatePreference({ push_enabled: value })}
              disabled={saving}
              trackColor={{ false: theme.colors.muted, true: theme.colors.primary500 }}
              thumbColor={preferences.push_enabled ? theme.colors.primary700 : theme.colors.surface200}
            />
          </Box>
        </Card>
        </Box>

        {/* Event Preferences */}
        <Box marginBottom="l">
        <Card variant="elevated" padding="l">
          <Box flexDirection="row" alignItems="center" marginBottom="m">
            <Box
              width={28}
              height={28}
              borderRadius="m"
              backgroundColor="primary500"
              marginRight="s"
              alignItems="center"
              justifyContent="center"
              style={theme.shadows.sm}
            >
              <Feather name="check-circle" size={14} color="#fff" />
            </Box>
            <Text variant="subtitle" color="text">
              Tipuri de evenimente
            </Text>
          </Box>

          <Text variant="caption" color="muted" marginBottom="m">
            Alege ce evenimente generează notificări
          </Text>

          {Object.entries(preferences.event_preferences).map(([eventType, channels], index) => (
            <Box
              key={eventType}
              paddingVertical="m"
              borderBottomWidth={index < Object.keys(preferences.event_preferences).length - 1 ? 1 : 0}
              borderBottomColor="border"
            >
              <Text variant="label" color="text" marginBottom="s">
                {formatEventName(eventType)}
              </Text>
              <Box flexDirection="row" columnGap="l">
                {/* Email checkbox */}
                <Pressable
                  onPress={() =>
                    handleEventPreferenceChange(eventType, 'email', !channels.email)
                  }
                  disabled={!preferences.email_enabled || saving}
                  style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}
                >
                  <Box
                    width={20}
                    height={20}
                    borderRadius="xs"
                    borderWidth={2}
                    borderColor={
                      channels.email && preferences.email_enabled
                        ? 'primary500'
                        : 'muted'
                    }
                    backgroundColor={
                      channels.email && preferences.email_enabled
                        ? 'primary500'
                        : 'transparent'
                    }
                    alignItems="center"
                    justifyContent="center"
                    marginRight="xs"
                  >
                    {channels.email && preferences.email_enabled && (
                      <Feather name="check" size={12} color="#fff" />
                    )}
                  </Box>
                  <Text
                    variant="caption"
                    color={preferences.email_enabled ? 'text' : 'muted'}
                  >
                    Email
                  </Text>
                </Pressable>

                {/* Push checkbox */}
                <Pressable
                  onPress={() =>
                    handleEventPreferenceChange(eventType, 'push', !channels.push)
                  }
                  disabled={!preferences.push_enabled || saving}
                  style={{ flexDirection: 'row', alignItems: 'center', flex: 1 }}
                >
                  <Box
                    width={20}
                    height={20}
                    borderRadius="xs"
                    borderWidth={2}
                    borderColor={
                      channels.push && preferences.push_enabled
                        ? 'primary500'
                        : 'muted'
                    }
                    backgroundColor={
                      channels.push && preferences.push_enabled
                        ? 'primary500'
                        : 'transparent'
                    }
                    alignItems="center"
                    justifyContent="center"
                    marginRight="xs"
                  >
                    {channels.push && preferences.push_enabled && (
                      <Feather name="check" size={12} color="#fff" />
                    )}
                  </Box>
                  <Text
                    variant="caption"
                    color={preferences.push_enabled ? 'text' : 'muted'}
                  >
                    Push
                  </Text>
                </Pressable>
              </Box>
            </Box>
          ))}
        </Card>
        </Box>

        {/* Severity Filter */}
        <Box marginBottom="l">
        <Card variant="elevated" padding="l">
          <Box flexDirection="row" alignItems="center" marginBottom="m">
            <Box
              width={28}
              height={28}
              borderRadius="m"
              backgroundColor="warning"
              marginRight="s"
              alignItems="center"
              justifyContent="center"
              style={theme.shadows.sm}
            >
              <Feather name="alert-triangle" size={14} color="#fff" />
            </Box>
            <Text variant="subtitle" color="text">
              Severitate minimă
            </Text>
          </Box>

          <Text variant="caption" color="muted" marginBottom="m">
            Filtrează notificările pe baza severității
          </Text>

          <Box flexDirection="row" columnGap="s">
            {(['low', 'medium', 'high'] as const).map((severity) => (
              <Pressable
                key={severity}
                onPress={() => updatePreference({ min_severity: severity })}
                disabled={saving}
                style={{ flex: 1 }}
              >
                <Box
                  padding="m"
                  borderRadius="m"
                  borderWidth={2}
                  borderColor={
                    preferences.min_severity === severity ? 'primary500' : 'border'
                  }
                  backgroundColor={
                    preferences.min_severity === severity ? 'primary100' : 'card'
                  }
                  alignItems="center"
                >
                  <Text
                    variant="label"
                    color={preferences.min_severity === severity ? 'primary700' : 'text'}
                    style={{ textTransform: 'capitalize' }}
                  >
                    {severity === 'low' ? 'Scăzută' : severity === 'medium' ? 'Medie' : 'Înaltă'}
                  </Text>
                </Box>
              </Pressable>
            ))}
          </Box>
        </Card>
        </Box>

        {/* Quiet Hours */}
        <Box marginBottom="xl">
        <Card variant="elevated" padding="l">
          <Box flexDirection="row" alignItems="center" marginBottom="m">
            <Box
              width={28}
              height={28}
              borderRadius="m"
              backgroundColor="secondary500"
              marginRight="s"
              alignItems="center"
              justifyContent="center"
              style={theme.shadows.sm}
            >
              <Feather name="moon" size={14} color="#fff" />
            </Box>
            <Text variant="subtitle" color="text">
              Ore de liniște
            </Text>
          </Box>

          <Box
            flexDirection="row"
            alignItems="center"
            justifyContent="space-between"
            marginBottom="m"
          >
            <Box flex={1}>
              <Text variant="label" color="text" marginBottom="xs">
                Activează ore de liniște
              </Text>
              <Text variant="caption" color="muted">
                Dezactivează notificările în intervalul specificat
              </Text>
            </Box>
            <Switch
              value={preferences.quiet_hours?.enabled || false}
              onValueChange={(value) =>
                updatePreference({
                  quiet_hours: {
                    ...preferences.quiet_hours,
                    enabled: value,
                  },
                })
              }
              disabled={saving}
              trackColor={{ false: theme.colors.muted, true: theme.colors.primary500 }}
              thumbColor={
                preferences.quiet_hours?.enabled
                  ? theme.colors.primary700
                  : theme.colors.surface200
              }
            />
          </Box>

          {preferences.quiet_hours?.enabled && (
            <Box
              padding="m"
              borderRadius="m"
              backgroundColor="surface50"
              borderWidth={1}
              borderColor="border"
            >
              <Text variant="caption" color="muted" marginBottom="m">
                Interval de liniște (UTC):
              </Text>

              <Box flexDirection="row" alignItems="center" justifyContent="space-between" gap="m">
                {/* Start Time Picker */}
                <Box flex={1}>
                  <Text variant="caption" color="muted" marginBottom="xs">
                    Ora de start
                  </Text>
                  <Pressable
                    onPress={() => setShowStartPicker(true)}
                    disabled={saving}
                  >
                    <Box
                      padding="m"
                      borderRadius="s"
                      borderWidth={1}
                      borderColor="border"
                      backgroundColor="card"
                      flexDirection="row"
                      alignItems="center"
                      justifyContent="center"
                    >
                      <Feather name="clock" size={16} color={theme.colors.primary500} style={{ marginRight: 8 }} />
                      <Text variant="body" color="text">
                        {preferences.quiet_hours?.start || '22:00'}
                      </Text>
                    </Box>
                  </Pressable>
                </Box>

                {/* Separator */}
                <Box marginTop="l">
                  <Text variant="body" color="muted">—</Text>
                </Box>

                {/* End Time Picker */}
                <Box flex={1}>
                  <Text variant="caption" color="muted" marginBottom="xs">
                    Ora de final
                  </Text>
                  <Pressable
                    onPress={() => setShowEndPicker(true)}
                    disabled={saving}
                  >
                    <Box
                      padding="m"
                      borderRadius="s"
                      borderWidth={1}
                      borderColor="border"
                      backgroundColor="card"
                      flexDirection="row"
                      alignItems="center"
                      justifyContent="center"
                    >
                      <Feather name="clock" size={16} color={theme.colors.primary500} style={{ marginRight: 8 }} />
                      <Text variant="body" color="text">
                        {preferences.quiet_hours?.end || '08:00'}
                      </Text>
                    </Box>
                  </Pressable>
                </Box>
              </Box>

              {/* Time Pickers */}
              {showStartPicker && (
                <DateTimePicker
                  value={parseTimeString(preferences.quiet_hours?.start || '22:00')}
                  mode="time"
                  is24Hour={true}
                  onChange={handleStartTimeChange}
                />
              )}
              {showEndPicker && (
                <DateTimePicker
                  value={parseTimeString(preferences.quiet_hours?.end || '08:00')}
                  mode="time"
                  is24Hour={true}
                  onChange={handleEndTimeChange}
                />
              )}
            </Box>
          )}
        </Card>
        </Box>

        <Box height={theme.spacing.l} />
      </Box>
    </Screen>
  );
}