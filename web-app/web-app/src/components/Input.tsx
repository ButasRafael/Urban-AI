import {
  forwardRef,
  useId,
  useRef,
  useState,
  type InputHTMLAttributes,
  type ReactNode,
  type Ref,
  type RefCallback,
  type MutableRefObject,
} from 'react';
import clsx from 'clsx';
import '../styles/input.css';

type Size = 'sm' | 'md' | 'lg';

interface Props extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  hint?: string;
  error?: string;
  size?: Size;
  fullWidth?: boolean;
  prefixIcon?: ReactNode;
  suffixIcon?: ReactNode;
  loading?: boolean;
  clearable?: boolean;
  revealToggle?: boolean;
  stopGlobalKeys?: boolean;
}

function mergeRefs<T>(...refs: Array<Ref<T> | undefined>): RefCallback<T> {
  return (value) => {
    for (const ref of refs) {
      if (!ref) continue;
      if (typeof ref === 'function') {
        ref(value);
      } else {
        (ref as MutableRefObject<T | null>).current = value;
      }
    }
  };
}

const Input = forwardRef<HTMLInputElement, Props>(function Input(
  {
    label,
    hint,
    error,
    size = 'md',
    fullWidth,
    prefixIcon,
    suffixIcon,
    loading,
    clearable,
    revealToggle,
    className,
    id,
    type = 'text',
    onChange,
    stopGlobalKeys = false,
    onKeyDown: userOnKeyDown,
    onKeyUp: userOnKeyUp,
    ...rest
  },
  ref
) {
  const autoId = useId();
  const inputId = id ?? `in-${autoId}`;
  const hintId = hint ? `${inputId}-hint` : undefined;
  const errId = error ? `${inputId}-err` : undefined;
  const describedBy = [errId, hintId].filter(Boolean).join(' ') || undefined;

  const localRef = useRef<HTMLInputElement>(null);
  const setInputRef = mergeRefs<HTMLInputElement>(ref, localRef);

  const [revealed, setRevealed] = useState(false);
  const isPassword = type === 'password';
  const showReveal = !!(revealToggle && isPassword);
  const actualType = showReveal ? (revealed ? 'text' : 'password') : type;

  const hasPrefix = !!prefixIcon;
  const hasSuffix = !!suffixIcon || !!loading || !!clearable || !!showReveal;

  function handleClear() {
    const el = localRef.current;
    if (!el) return;
    el.value = '';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.focus();
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (stopGlobalKeys) e.stopPropagation();
    userOnKeyDown?.(e);
  };
  const handleKeyUp = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (stopGlobalKeys) e.stopPropagation();
    userOnKeyUp?.(e);
  };

  const bare =
    !label && !hint && !error && !hasPrefix && !hasSuffix && !fullWidth && size === 'md';

  if (bare) {
    return (
      <input
        {...rest}
        id={inputId}
        ref={setInputRef}
        type={actualType}
        aria-invalid={!!error || undefined}
        aria-describedby={describedBy}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        className={clsx('input', className)}
      />
    );
  }

  return (
    <div className={clsx('form-control', fullWidth && 'is-block')}>
      {label && <label htmlFor={inputId} className="form-label">{label}</label>}

      <div
        className={clsx('input-wrap', `input-${size}`, error && 'is-invalid')}
        data-has-prefix={hasPrefix || undefined}
        data-has-suffix={hasSuffix || undefined}
      >
        {hasPrefix && <span className="input-affix -prefix">{prefixIcon}</span>}

        <input
          {...rest}
          id={inputId}
          ref={setInputRef}
          type={actualType}
          aria-invalid={!!error || undefined}
          aria-describedby={describedBy}
          onChange={onChange}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          className="input"
        />

        {(hasSuffix || loading || showReveal || clearable) && (
          <span className="input-affix -suffix">
            {suffixIcon}
            {loading && (
              <span className="in-spinner" aria-hidden>
                <svg viewBox="0 0 24 24" width="18" height="18">
                  <circle className="ring" cx="12" cy="12" r="9" />
                  <circle className="bar"  cx="12" cy="12" r="9" />
                </svg>
              </span>
            )}
            {showReveal && (
              <button
                type="button"
                className="affix-btn"
                tabIndex={-1}
                onClick={() => setRevealed(v => !v)}
                aria-label={revealed ? 'Hide password' : 'Show password'}
                title={revealed ? 'Hide password' : 'Show password'}
              >
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
                  {revealed
                    ? (<><path d="M17.94 17.94A10.94 10.94 0 0 1 12 20C7 20 2.73 16.11 1 12c.62-1.45 1.52-2.79 2.63-3.96"/><path d="M9.9 4.24A10.94 10.94 0 0 1 12 4c5 0 9.27 3.89 11 8a11.66 11.66 0 0 1-2.1 3.2"/><path d="M3 3l18 18"/></>)
                    : (<><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12Z" /><circle cx="12" cy="12" r="3" /></>)}
                </svg>
              </button>
            )}
            {clearable && (
              <button
                type="button"
                className="affix-btn"
                tabIndex={-1}
                onClick={handleClear}
                aria-label="Clear"
                title="Clear"
              >
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            )}
          </span>
        )}
      </div>

      {error ? (
        <div id={errId} className="form-error">{error}</div>
      ) : (
        hint && <div id={hintId} className="form-hint">{hint}</div>
      )}
    </div>
  );
});

export default Input;
