// src/screens/NotificationsScreen.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  SectionList,
  RefreshControl,
  Pressable,
  View,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { Feather } from '@expo/vector-icons';
import { useTheme } from '@shopify/restyle';

import { RootStackParamList } from '../navigation/types';
import Screen from '../components/ui/Screen';
import HeroHeader from '../components/ui/HeroHeader';
import Card from '../components/ui/Card';
import StyledButton from '../components/StyledButton';
import Badge from '../components/ui/Badge';
import { Box, Text } from '../components/restylePrimitives';
import type { Theme } from '../theme';
import { notify } from '../components/ui/Toast';

import {
  getNotifications,
  markNotificationAsRead,
  markAllNotificationsAsRead,
  deleteNotification,
  type StoredNotification,
} from '../api/notifications';
import { useNotifications } from '../contexts/NotificationContext';

type Props = NativeStackScreenProps<RootStackParamList, 'Notifications'>;

type MobileGroup = { title: string; data: StoredNotification[] };

const PAGE_SIZE = 50 as const;

export default function NotificationsScreen({}: Props) {
  const theme = useTheme<Theme>();

  // state
  const [items, setItems] = useState<StoredNotification[]>([]);
  const [filter, setFilter] = useState<'all' | 'unread'>('all');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const offsetRef = useRef(0);

  const { notificationReceived, refreshUnreadCount } = useNotifications();

  // core loader
  const loadNotifications = useCallback(
    async (append = false) => {
      try {
        if (!append) {
          if (!refreshing) setLoading(true);
          offsetRef.current = 0;
        } else {
          setLoadingMore(true);
        }

        const currentOffset = append ? offsetRef.current : 0;
        const data = await getNotifications(PAGE_SIZE, currentOffset, filter === 'unread');

        setItems(prev => {
          if (append) {
            // de-dupe by id in case of overlap
            const existing = new Set(prev.map(n => n.id));
            const newOnes = data.filter(n => !existing.has(n.id));
            return [...prev, ...newOnes];
          }
          return data;
        });

        setHasMore(data.length === PAGE_SIZE);
        offsetRef.current = (append ? offsetRef.current : 0) + data.length;
      } catch (e) {
        notify.error('Eroare', 'Nu s-au putut încărca notificările');
      } finally {
        setLoading(false);
        setLoadingMore(false);
        setRefreshing(false);
      }
    },
    [filter, refreshing]
  );

  // initial + filter changes
  useEffect(() => {
    loadNotifications(false);
  }, [filter, loadNotifications]);

  // live updates via WebSocket — refetch list when one arrives
  useEffect(() => {
    if (notificationReceived > 0) {
      loadNotifications(false);
    }
  }, [notificationReceived, loadNotifications]);

  // actions
  const onMarkOne = async (id: number) => {
    try {
      await markNotificationAsRead(id);
      setItems(prev => prev.map(n => (n.id === id ? { ...n, read_at: new Date().toISOString() } : n)));
      refreshUnreadCount();
    } catch {
      notify.error('Eroare', 'Nu s-a putut marca ca citit');
    }
  };

  const onMarkAll = async () => {
    try {
      await markAllNotificationsAsRead();
      setItems(prev => prev.map(n => ({ ...n, read_at: new Date().toISOString() })));
      refreshUnreadCount();
      notify.success('Gata', 'Toate notificarile au fost marcate ca citite');
    } catch {
      notify.error('Eroare', 'Acțiunea a eșuat');
    }
  };

  const onDelete = async (id: number) => {
    try {
      await deleteNotification(id);
      setItems(prev => prev.filter(n => n.id !== id));
      refreshUnreadCount();
      notify.info('Șters', 'Notificarea a fost ștearsă');
    } catch {
      notify.error('Eroare', 'Nu s-a putut șterge notificarea');
    }
  };

  const unreadCount = useMemo(() => items.filter(n => !n.read_at).length, [items]);

  // grouping & sections
  const sections: MobileGroup[] = useMemo(() => {
    const grouped = groupNotificationsByDate(items);
    return grouped.map(g => ({ title: g.label, data: g.notifications }));
  }, [items]);

  // list footer
  const ListFooter = () => {
    if (!hasMore) return null;
    return (
      <Box padding="l" alignItems="center">
        <StyledButton
          title={loadingMore ? 'Se încarcă…' : 'Încarcă mai multe'}
          onPress={() => loadNotifications(true)}
          loading={loadingMore}
          variant="ghost"
        />
      </Box>
    );
  };

  // UI
  if (loading && !refreshing) {
    return (
      <Screen scroll padded={false} edges={['left', 'right']} center maxWidth={640}>
        <HeroHeader
          title="Notificări"
          subtitle="Inbox"
          rightSlot={
            <Box
              width={56}
              height={56}
              borderRadius="lg"
              alignItems="center"
              justifyContent="center"
              style={{
                backgroundColor: 'rgba(255,255,255,0.15)',
                borderWidth: 1,
                borderColor: 'rgba(255,255,255,0.25)',
                ...theme.shadows.sm,
              }}
            >
              <Feather name="bell" size={24} color="#fff" />
            </Box>
          }
          height={180}
          blobs
        />
        <Box paddingHorizontal="l" style={{ marginTop: -32 }}>
          <Box flex={1} alignItems="center" justifyContent="center" padding="l" style={{ minHeight: 300 }}>
            <ActivityIndicator size="large" color={theme.colors.primary500} />
          </Box>
        </Box>
      </Screen>
    );
  }

  return (
    <Screen scroll padded={false} edges={['left', 'right']} center maxWidth={640} dismissKeyboardOnTap>
      <HeroHeader
        title="Notificări"
        subtitle="Inbox"
        rightSlot={
          <Box
            width={56}
            height={56}
            borderRadius="lg"
            alignItems="center"
            justifyContent="center"
            style={{
              backgroundColor: 'rgba(255,255,255,0.15)',
              borderWidth: 1,
              borderColor: 'rgba(255,255,255,0.25)',
              ...theme.shadows.sm,
            }}
          >
            <Feather name="bell" size={24} color="#fff" />
          </Box>
        }
        height={180}
        blobs
      />

      <Box paddingHorizontal="l" style={{ marginTop: -32 }}>
        {/* header row: filters + mark all */}
        <Box flexDirection="row" alignItems="center" justifyContent="space-between" marginBottom="m" gap="s">
          <Box flexDirection="row" alignItems="center" flex={1}>
            <Text variant="title" color="text" style={{ fontWeight: '800' }}>
              Inbox
            </Text>
            {unreadCount > 0 && (
              <Badge variant="primary" size="sm" style={{ marginLeft: 8 }}>
                {unreadCount}
              </Badge>
            )}
          </Box>

          {unreadCount > 0 && (
            <Pressable
              onPress={onMarkAll}
              hitSlop={10}
              style={({ pressed }) => ({
                paddingHorizontal: 12,
                paddingVertical: 8,
                borderRadius: theme.borderRadii.m,
                backgroundColor: pressed ? theme.colors.primary100 : 'transparent',
                flexDirection: 'row',
                alignItems: 'center',
                gap: 6,
              })}
            >
              <Feather name="check-square" size={18} color={theme.colors.primary500} />
              <Text variant="label" color="primary500" style={{ fontWeight: '600' }}>
                Tot
              </Text>
            </Pressable>
          )}
        </Box>

        {/* segmented control */}
        <Segmented
          value={filter}
          onChange={setFilter}
          options={[
            { key: 'all', label: 'Toate' },
            { key: 'unread', label: 'Necitite' },
          ]}
        />

        {/* empty state */}
        {items.length === 0 ? (
          <Card variant="elevated" padding="l" style={{ marginTop: theme.spacing.l }}>
            <Box alignItems="center">
              <Box
                width={64}
                height={64}
                borderRadius="lg"
                backgroundColor="primary100"
                alignItems="center"
                justifyContent="center"
                style={{ marginBottom: theme.spacing.m }}
              >
                <Feather name="bell" size={28} color={theme.colors.primary700} />
              </Box>
              <Text variant="title" color="text" style={{ fontWeight: '700' }} marginBottom="s">
                {filter === 'unread' ? 'Ești la zi!' : 'Nu sunt notificări'}
              </Text>
              <Text variant="body" color="muted" style={{ textAlign: 'center' }}>
                {filter === 'unread'
                  ? 'Nu ai notificări necitite în acest moment.'
                  : 'Notificările tale vor apărea aici.'}
              </Text>
            </Box>
          </Card>
        ) : (
          <Box marginTop="m">
            {sections.map((section, sectionIndex) => (
              <Box key={section.title} marginBottom="l">
                {/* Section header */}
                <Box marginBottom="s">
                  <Box
                    alignSelf="flex-start"
                    paddingHorizontal="m"
                    paddingVertical="xs"
                    borderRadius="pill"
                    backgroundColor="primary100"
                    borderWidth={1}
                    borderColor="primary300"
                  >
                    <Text variant="label" color="primary700" style={{ textTransform: 'uppercase', letterSpacing: 1 }}>
                      {section.title}
                    </Text>
                  </Box>
                </Box>

                {/* Section items */}
                {section.data.map((item, index) => (
                  <Box key={item.id} marginBottom="s">
                    <NotificationRow
                      item={item}
                      onMarkAsRead={onMarkOne}
                      onDelete={onDelete}
                    />
                  </Box>
                ))}
              </Box>
            ))}

            {/* Load more footer */}
            {hasMore && (
              <Box padding="l" alignItems="center">
                <StyledButton
                  title={loadingMore ? 'Se încarcă…' : 'Încarcă mai multe'}
                  onPress={() => loadNotifications(true)}
                  loading={loadingMore}
                  variant="ghost"
                />
              </Box>
            )}
          </Box>
        )}

        <Box height={theme.spacing.xl} />
      </Box>
    </Screen>
  );
}

/* ----------------------------- UI bits ----------------------------- */

function Segmented({
  value,
  onChange,
  options,
}: {
  value: 'all' | 'unread';
  onChange: (v: 'all' | 'unread') => void;
  options: { key: 'all' | 'unread'; label: string }[];
}) {
  const theme = useTheme<Theme>();
  return (
    <Box
      flexDirection="row"
      padding="xs"
      borderRadius="lg"
      backgroundColor="card"
      borderWidth={1}
      borderColor="border"
      style={[theme.shadows.sm]}
    >
      {options.map(opt => {
        const active = value === opt.key;
        return (
          <Pressable
            key={opt.key}
            onPress={() => onChange(opt.key)}
            style={{
              flex: 1,
              paddingVertical: 8,
              borderRadius: theme.borderRadii.lg,
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: active ? theme.colors.primary100 : 'transparent',
            }}
          >
            <Text
              variant="label"
              color={active ? 'primary700' : 'text'}
              style={{ fontWeight: active ? '800' : '600' }}
            >
              {opt.label}
            </Text>
          </Pressable>
        );
      })}
    </Box>
  );
}

function NotificationRow({
  item,
  onMarkAsRead,
  onDelete,
}: {
  item: StoredNotification;
  onMarkAsRead: (id: number) => void;
  onDelete: (id: number) => void;
}) {
  const theme = useTheme<Theme>();
  const isUnread = !item.read_at;
  const ts = formatTimestamp(item.created_at);
  const message = getNotificationMessage(item);
  const iconDef = getNotificationIcon(item.event_type, theme);

  return (
    <Card
      variant="elevated"
      padding="m"
      style={{
        borderLeftWidth: isUnread ? 3 : 1,
        borderLeftColor: isUnread ? theme.colors.primary500 : theme.colors.border,
      }}
    >
      <Box flexDirection="row" alignItems="flex-start">
        {/* Icon bubble */}
        <Box
          width={40}
          height={40}
          borderRadius="m"
          alignItems="center"
          justifyContent="center"
          style={{
            backgroundColor: iconDef.bg,
            borderWidth: 1,
            borderColor: iconDef.border,
            marginRight: 12,
          }}
        >
          <Feather name={iconDef.name} size={18} color={iconDef.color} />
        </Box>

        {/* Content */}
        <Box flex={1}>
          <Box flexDirection="row" alignItems="center" justifyContent="space-between">
            <Text variant="label" color="primary700" style={{ textTransform: 'capitalize' }}>
              {item.event_type.replace(/_/g, ' ')}
            </Text>
            <Text variant="caption" color="muted">
              {ts}
            </Text>
          </Box>

          <Text variant="body" color="text" style={{ marginTop: 4 }}>
            {message}
          </Text>

          {/* channel tag */}
          {item.channel ? (
            <Badge
              variant={item.channel === 'email' ? 'secondary' : 'primary'}
              size="sm"
              style={{ marginTop: 8 }}
            >
              {item.channel === 'email' ? 'Email trimis' : 'Push notification'}
            </Badge>
          ) : null}
        </Box>

        {/* Actions */}
        <Box flexDirection="row" marginLeft="s">
          {isUnread && (
            <IconButton
              icon="check"
              onPress={() => onMarkAsRead(item.id)}
              accessibilityLabel="Marchează citit"
            />
          )}
          <IconButton
            icon="trash-2"
            onPress={() => onDelete(item.id)}
            danger
            accessibilityLabel="Șterge notificarea"
          />
        </Box>
      </Box>
    </Card>
  );
}

function IconButton({
  icon,
  onPress,
  danger,
  accessibilityLabel,
}: {
  icon: React.ComponentProps<typeof Feather>['name'];
  onPress: () => void;
  danger?: boolean;
  accessibilityLabel?: string;
}) {
  const theme = useTheme<Theme>();
  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel}
      style={{
        width: 36,
        height: 36,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 10,
        borderWidth: 1,
        borderColor: danger ? theme.colors.error : theme.colors.border,
        marginLeft: 6,
      }}
    >
      <Feather
        name={icon}
        size={16}
        color={danger ? theme.colors.error : theme.colors.text}
      />
    </Pressable>
  );
}

/* ------------------------- helpers / logic ------------------------- */

function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();

  const mins = Math.floor(diffMs / 60000);
  const hours = Math.floor(diffMs / 3600000);
  const days = Math.floor(diffMs / 86400000);

  if (mins < 1) return 'Chiar acum';
  if (mins < 60) return `${mins}m`;
  if (hours < 24) return `${hours}h`;
  if (days < 7) return `${days}z`;
  return date.toLocaleDateString();
}

function getNotificationMessage(n: StoredNotification): string {
  const d = n.event_data || {};
  switch (n.event_type) {
    case 'media_created':
      return `Media nouă încărcată (${d.num_detections ?? 0} detecții)`;
    case 'issue_verified':
      return `Raportul tău ${d.class_name ?? ''} a fost verificat`;
    case 'issue_assigned':
      return `Problema ${d.class_name ?? ''} a fost asignată`;
    case 'issue_status_changed':
      return `${d.class_name ?? 'Problema'}: status ${d.old_status ?? '?'} → ${d.new_status ?? '?'}`;
    case 'issue_severity_changed':
      return `${d.class_name ?? 'Problema'}: severitate ${d.old_severity ?? '?'} → ${d.new_severity ?? '?'}`;
    default:
      return `Notificare: ${n.event_type}`;
  }
}

function getNotificationIcon(
  eventType: string,
  theme: Theme
): { name: React.ComponentProps<typeof Feather>['name']; color: string; bg: string; border: string } {
  switch (eventType) {
    case 'media_created':
      return {
        name: 'upload-cloud',
        color: theme.colors.primary700,
        bg: theme.colors.primary100,
        border: theme.colors.primary300,
      };
    case 'issue_verified':
      return {
        name: 'check-circle',
        color: theme.colors.success,
        bg: theme.colors.surface50,
        border: theme.colors.success,
      };
    case 'issue_assigned':
      return {
        name: 'user-check',
        color: theme.colors.primary700,
        bg: theme.colors.primary100,
        border: theme.colors.primary300,
      };
    case 'issue_status_changed':
      return {
        name: 'activity',
        color: theme.colors.secondary500,
        bg: theme.colors.primary100,
        border: theme.colors.primary300,
      };
    case 'issue_severity_changed':
      return {
        name: 'alert-triangle',
        color: theme.colors.warning,
        bg: theme.colors.surface50,
        border: theme.colors.warning,
      };
    default:
      return {
        name: 'bell',
        color: theme.colors.primary700,
        bg: theme.colors.primary100,
        border: theme.colors.primary300,
      };
  }
}

function groupNotificationsByDate(notifs: StoredNotification[]): { label: string; notifications: StoredNotification[] }[] {
  const groups: Record<string, StoredNotification[]> = {};
  const now = new Date();

  for (const n of notifs) {
    const d = new Date(n.created_at);
    const key = getDateGroupKey(d, now);
    (groups[key] ??= []).push(n);
  }

  const order = ['Today', 'Yesterday', 'This Week', 'This Month'];
  const out: { label: string; notifications: StoredNotification[] }[] = [];

  for (const k of order) {
    if (groups[k]) {
      out.push({ label: translateGroupLabel(k), notifications: groups[k] });
      delete groups[k];
    }
  }

  // remaining groups by recency (month-year)
  Object.entries(groups)
    .sort(([, a], [, b]) => new Date(b[0].created_at).getTime() - new Date(a[0].created_at).getTime())
    .forEach(([label, notifications]) => {
      out.push({ label, notifications });
    });

  return out;
}

function getDateGroupKey(date: Date, now: Date): string {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startOfYesterday = new Date(startOfToday); startOfYesterday.setDate(startOfYesterday.getDate() - 1);
  const startOfThisWeek = new Date(startOfToday); startOfThisWeek.setDate(startOfThisWeek.getDate() - 7);
  const startOfThisMonth = new Date(startOfToday); startOfThisMonth.setDate(startOfThisMonth.getDate() - 30);

  const onlyDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());

  if (onlyDate.getTime() >= startOfToday.getTime()) return 'Today';
  if (onlyDate.getTime() >= startOfYesterday.getTime()) return 'Yesterday';
  if (onlyDate.getTime() >= startOfThisWeek.getTime()) return 'This Week';
  if (onlyDate.getTime() >= startOfThisMonth.getTime()) return 'This Month';
  return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
}

function translateGroupLabel(key: string): string {
  switch (key) {
    case 'Today': return 'Astăzi';
    case 'Yesterday': return 'Ieri';
    case 'This Week': return 'Săptămâna aceasta';
    case 'This Month': return 'Luna aceasta';
    default: return key;
  }
}
