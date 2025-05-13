import type { InputHTMLAttributes } from 'react';
export default function Input(p: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...p} className={clsx('input', p.className)} />;
}
import clsx from 'clsx';
