import type { ReactNode } from 'react';
import { Navigate } from 'react-router-dom';
import { useSession } from './SessionProvider';

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { session, loading } = useSession();
  if (loading) return <p style={{ padding: 24 }}>Loading…</p>;
  if (!session) return <Navigate to="/login" replace />;
  return <>{children}</>;
}
