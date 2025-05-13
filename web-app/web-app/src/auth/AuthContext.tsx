import {
  createContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import { jwtDecode } from 'jwt-decode';
import client from '../api/client';

export type UserRole = 'user' | 'authority' | 'admin';
export interface User { username: string; role: UserRole; }

interface AuthCtx {
  user: User | null;
  setUser(u: User | null): void;
  loading: boolean;
}

export const AuthContext = createContext<AuthCtx | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    const tok = localStorage.getItem('accessToken');
    try { return tok ? (jwtDecode(tok) as User) : null; }
    catch { return null; }
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const tok = localStorage.getItem('accessToken');
    if (!tok) {
        setLoading(false);
        return;
    }

    client
        .get<User>('/auth/me')
        .then((r) => setUser(r.data))
        .catch(() => {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        })
        .finally(() => setLoading(false));
    }, []);

  return (
    <AuthContext.Provider value={{ user, setUser, loading }}>
      {children}
    </AuthContext.Provider>
  );
}