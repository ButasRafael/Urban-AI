import type { ButtonHTMLAttributes, ReactNode } from 'react';
import clsx from 'clsx';

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger';

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

export default function Button({ variant = 'primary', className, ...rest }: Props) {
  return (
    <button
      {...rest}
      className={clsx('btn', `btn-${variant}`, className)}
    />
  );
}
