import type { Problem } from '../api/problems';

interface Props {
  problem: Problem | null;
  onClose(): void;
}

export default function IssueModal({ problem, onClose }: Props) {
  if (!problem) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
      }}
      onClick={onClose}
    >
      <div
        style={{
          background: 'var(--surface)',
          padding: 'var(--l)',
          borderRadius: 'var(--radius)',
          minWidth: 300,
          boxShadow: '0 4px 10px rgba(0,0,0,.15)',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2
          style={{
            marginBottom: 'var(--s)',
            fontSize: 18,
            fontWeight: 600,
            color: 'var(--primary)',
          }}
        >
          Issue #{problem.media_id}
        </h2>

        <p>
          <strong>Date:</strong>{' '}
          {new Date(problem.created_at).toLocaleString()}
        </p>
        <p>
          <strong>User:</strong> {problem.user_username}
        </p>
        <p>
          <strong>Address:</strong> {problem.address ?? '-'}
        </p>
        <p>
          <strong>Classes:</strong>{' '}
          {problem.predicted_classes.join(', ') || '-'}
        </p>

        <button
          onClick={onClose}
          className="btn btn-primary"
          style={{ marginTop: 'var(--m)', width: '100%' }}
        >
          Close
        </button>
      </div>
    </div>
  );
}
