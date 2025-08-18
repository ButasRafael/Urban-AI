import type { ButtonHTMLAttributes, ReactNode } from 'react';
import clsx from 'clsx';
import '../styles/button.css';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';
type Size = 'sm' | 'md' | 'lg';

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  fullWidth?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
  loading?: boolean;
  iconOnly?: boolean;
  children?: ReactNode;
}

export default function Button({
  variant = 'primary',
  size = 'md',
  fullWidth,
  leftIcon,
  rightIcon,
  loading,
  iconOnly,
  className,
  children,
  disabled,
  type,
  ...rest
}: Props) {
  const isDisabled = disabled || loading;

  // A11y: ensure icon-only buttons have a name
  const ariaLabel =
    iconOnly && !rest['aria-label']
      ? (rest.title as string) || 'Button'
      : rest['aria-label'];

  // If iconOnly and you passed the SVG as children, render it as an icon.
  const hasChildIcon = !!children && typeof children !== 'string';
  const renderLeftIcon =
    leftIcon ?? (iconOnly && hasChildIcon ? children : undefined);

  const renderLabel =
    !iconOnly || (typeof children === 'string' && children.trim().length > 0)
      ? children
      : null;

  return (
    <button
      {...rest}
      type={type ?? 'button'}
      aria-label={ariaLabel}
      disabled={isDisabled}
      aria-disabled={isDisabled || undefined}
      aria-busy={loading || undefined}
      aria-live={loading ? 'polite' : undefined}
      data-icon-only={iconOnly || undefined}
      data-variant={variant}
      data-size={size}
      data-loading={loading || undefined}
      className={clsx(
        'btn',
        `btn-${variant}`,
        `btn-${size}`,
        fullWidth && 'btn-block',
        isDisabled && 'is-disabled',
        className
      )}
    >
      {loading && (
        <span className="btn__spinner" aria-hidden>
          <svg viewBox="0 0 24 24" width="18" height="18">
            <circle className="ring" cx="12" cy="12" r="9" />
            <circle className="bar"  cx="12" cy="12" r="9" />
          </svg>
        </span>
      )}

      <span className={clsx('btn__inner', loading && 'btn__inner--loading')}>
        {renderLeftIcon && <span className="btn__icon -left" aria-hidden>{renderLeftIcon}</span>}
        {renderLabel && <span className="btn__label">{renderLabel}</span>}
        {rightIcon && <span className="btn__icon -right" aria-hidden>{rightIcon}</span>}
      </span>
    </button>
  );
}
