import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Protected from './components/ProtectedRoute';
import { NotificationProvider } from './contexts/NotificationContext';
import { useAuth } from './auth/useAuth';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import VerifyEmailPage from './pages/VerifyEmailPage';
import ForgotPasswordPage from './pages/ForgotPasswordPage';
import ResetPasswordPage from './pages/ResetPasswordPage';
import AnalyticsPage from './pages/AnalyticsPage';
import MapPage from './pages/MapPage';
import ListPage from './pages/ListPage';
import ChatPage from "./pages/ChatPage";
import IssuesPage from './pages/IssuesPage';
import NotificationSettings from './pages/NotificationSettings';
import Notifications from './pages/Notifications';

function LayoutWithNotifications() {
  const { user } = useAuth();
  const isAuthed = !!user;

  return (
    <NotificationProvider enabled={isAuthed}>
      <Layout />
    </NotificationProvider>
  );
}

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth pages – no shell, so they can be full-bleed */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/verify-email" element={<VerifyEmailPage />} />
        <Route path="/forgot-password" element={<ForgotPasswordPage />} />
        <Route path="/reset-password" element={<ResetPasswordPage />} />
        <Route path="/" element={<Landing />} />

        {/* App pages – wrapped by the shell with notification context */}
        <Route element={<LayoutWithNotifications />}>
          <Route path="/analytics" element={<Protected role="admin"><AnalyticsPage /></Protected>} />
          <Route path="/map"       element={<Protected><MapPage /></Protected>} />
          <Route path="/list"      element={<Protected><ListPage /></Protected>} />
          <Route path="/chat"      element={<Protected role="authority"><ChatPage /></Protected>} />
          <Route path="/issues"    element={<Protected><IssuesPage /></Protected>} />
          <Route path="/notifications" element={<Protected><Notifications /></Protected>} />
          <Route path="/settings/notifications" element={<Protected><NotificationSettings /></Protected>} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
