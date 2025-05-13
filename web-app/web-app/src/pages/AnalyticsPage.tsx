import { useEffect, useState } from 'react';
import {
  uploadsByDay,
  uploadsByUser,
  type DayStat,
  type UserStat,
} from '../api/analytics';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  ResponsiveContainer,
} from 'recharts';

export default function AnalyticsPage() {
  const [daily, setDaily] = useState<DayStat[]>([]);
  const [byUser, setByUser] = useState<UserStat[]>([]);

  useEffect(() => {
    uploadsByDay().then(setDaily);
    uploadsByUser().then(setByUser);
  }, []);

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', gap: 'var(--l)' }}>
      <h2 style={{ color: 'var(--primary)', marginBottom: 'var(--m)' }}>
        Uploads – last 7 days
      </h2>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={daily}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Line type="monotone" dataKey="count" stroke="var(--secondary)" />
        </LineChart>
      </ResponsiveContainer>

      <h2
        style={{
          color: 'var(--primary)',
          marginTop: 'var(--l)',
          marginBottom: 'var(--m)',
        }}
      >
        Uploads per user
      </h2>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={byUser}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="user" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="count" fill="var(--primary)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
