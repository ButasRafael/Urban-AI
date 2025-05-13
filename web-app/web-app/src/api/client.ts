import axios, { AxiosError } from "axios";
import type { AxiosRequestConfig } from "axios";

export const API_BASE = import.meta.env.VITE_API_BASE as string; 

const client = axios.create({
  baseURL: API_BASE,
  headers: { "Content-Type": "application/json" },
});

const ACCESS = "accessToken";
const REFRESH = "refreshToken";

const getLocal = (k: string) => window.localStorage.getItem(k);
const setLocal = (k: string, v: string) => window.localStorage.setItem(k, v);
const rmLocal = (...keys: string[]) => keys.forEach((k) => window.localStorage.removeItem(k));

client.interceptors.request.use((cfg) => {
  const t = getLocal(ACCESS);
  if (t) {
    if (cfg.headers) {
      cfg.headers['Authorization'] = `Bearer ${t}`;
    }
  }
  return cfg;
});

client.interceptors.response.use(
  (r) => r,
  async (err: AxiosError) => {
    const orig = err.config as (AxiosRequestConfig & { _retry?: true }) | undefined;

    if (err.response?.status === 401 && orig && !orig._retry) {
      orig._retry = true;
      try {
        const { data } = await axios.post(`${API_BASE}/auth/refresh`, {
          refresh_token: getLocal(REFRESH),
        });
        setLocal(ACCESS, data.access_token);
        setLocal(REFRESH, data.refresh_token);
        orig.headers = { ...orig.headers, Authorization: `Bearer ${data.access_token}` };
        return client(orig);
      } catch {
        rmLocal(ACCESS, REFRESH);
        //window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default client;