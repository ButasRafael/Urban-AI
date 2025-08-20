import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Protected from './components/ProtectedRoute';
import Landing from './pages/Landing';
import Login from './pages/Login';
import AnalyticsPage from './pages/AnalyticsPage';
import MapPage from './pages/MapPage';
import ListPage from './pages/ListPage';
import ChatPage from "./pages/ChatPage";
import IssuesPage from './pages/IssuesPage';
import Register from './pages/Register';

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Auth pages – no shell, so they can be full-bleed */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/" element={<Landing />} />

        {/* App pages – wrapped by the shell */}
        <Route element={<Layout />}>
          <Route path="/analytics" element={<Protected role="admin"><AnalyticsPage /></Protected>} />
          <Route path="/map"       element={<Protected><MapPage /></Protected>} />
          <Route path="/list"      element={<Protected><ListPage /></Protected>} />
          <Route path="/chat"      element={<Protected role="authority"><ChatPage /></Protected>} />
          <Route path="/issues"    element={<Protected><IssuesPage /></Protected>} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
