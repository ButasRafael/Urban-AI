import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Protected from './components/ProtectedRoute';
import Landing from './pages/Landing';
import Login from './pages/Login';
import AnalyticsPage from './pages/AnalyticsPage';
import MapPage from './pages/MapPage';
import ListPage from './pages/ListPage';
import ChatPage from "./pages/ChatPage";

export default function AppRouter() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/login" element={<Login />} />

          <Route
            path="/analytics"
            element={
              <Protected role="admin">
                <AnalyticsPage />
              </Protected>
            }
          />
          <Route
            path="/map"
            element={
              <Protected>
                <MapPage />
              </Protected>
            }
          />
          <Route
            path="/list"
            element={
              <Protected>
                <ListPage />
              </Protected>
            }
          />
          <Route
            path="/chat"
            element={
              <Protected role="authority">
                <ChatPage />
              </Protected>
            }
          />


          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
