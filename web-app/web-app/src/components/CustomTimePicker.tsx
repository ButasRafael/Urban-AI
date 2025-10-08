import { useState, useRef, useEffect } from 'react';
import '../styles/custom-time-picker.css';

interface CustomTimePickerProps {
  value: string; // Format: "HH:MM"
  onChange: (value: string) => void;
  disabled?: boolean;
  label?: string;
}

export default function CustomTimePicker({ value, onChange, disabled, label }: CustomTimePickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [hours, setHours] = useState(() => value.split(':')[0] || '00');
  const [minutes, setMinutes] = useState(() => value.split(':')[1] || '00');
  const [inputValue, setInputValue] = useState(value);
  const pickerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close picker when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  // Update internal state when value prop changes
  useEffect(() => {
    const [h, m] = value.split(':');
    if (h) setHours(h);
    if (m) setMinutes(m);
    setInputValue(value);
  }, [value]);

  function handleHourSelect(hour: string) {
    setHours(hour);
    const newTime = `${hour}:${minutes}`;
    setInputValue(newTime);
    onChange(newTime);
  }

  function handleMinuteSelect(minute: string) {
    setMinutes(minute);
    const newTime = `${hours}:${minute}`;
    setInputValue(newTime);
    onChange(newTime);
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const newValue = e.target.value;
    setInputValue(newValue);

    // Validate time format (HH:MM or H:MM)
    const timeRegex = /^([0-1]?[0-9]|2[0-3]):([0-5][0-9])$/;
    if (timeRegex.test(newValue)) {
      const [h, m] = newValue.split(':');
      const paddedHours = h.padStart(2, '0');
      const paddedMinutes = m.padStart(2, '0');
      setHours(paddedHours);
      setMinutes(paddedMinutes);
      onChange(`${paddedHours}:${paddedMinutes}`);
    }
  }

  function handleInputBlur() {
    // Validate and format on blur
    const timeRegex = /^([0-1]?[0-9]|2[0-3]):([0-5][0-9])$/;
    if (timeRegex.test(inputValue)) {
      const [h, m] = inputValue.split(':');
      const paddedHours = h.padStart(2, '0');
      const paddedMinutes = m.padStart(2, '0');
      const formattedTime = `${paddedHours}:${paddedMinutes}`;
      setInputValue(formattedTime);
      setHours(paddedHours);
      setMinutes(paddedMinutes);
      onChange(formattedTime);
    } else {
      // Reset to last valid value if invalid
      setInputValue(`${hours}:${minutes}`);
    }
  }

  function handleInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      handleInputBlur();
      inputRef.current?.blur();
    } else if (e.key === 'Escape') {
      setInputValue(`${hours}:${minutes}`);
      inputRef.current?.blur();
      setIsOpen(false);
    }
  }

  const hoursArray = Array.from({ length: 24 }, (_, i) => i.toString().padStart(2, '0'));
  const minutesArray = Array.from({ length: 12 }, (_, i) => (i * 5).toString().padStart(2, '0'));

  return (
    <div className="custom-time-picker" ref={pickerRef}>
      {label && <label className="time-picker-label" onClick={() => inputRef.current?.focus()}>{label}</label>}

      <div className={`time-picker-input ${isOpen ? 'active' : ''} ${disabled ? 'disabled' : ''}`}>
        <svg className="time-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
        <input
          ref={inputRef}
          type="text"
          className="time-display"
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyDown={handleInputKeyDown}
          disabled={disabled}
          placeholder="HH:MM"
          maxLength={5}
        />
        <button
          type="button"
          className="chevron-button"
          onClick={() => !disabled && setIsOpen(!isOpen)}
          disabled={disabled}
          tabIndex={-1}
        >
          <svg
            className={`chevron ${isOpen ? 'open' : ''}`}
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </button>
      </div>

      {isOpen && (
        <div className="time-picker-dropdown">
          <div className="time-picker-panel">
            <div className="time-column">
              <div className="column-header">Hours</div>
              <div className="time-scroll">
                {hoursArray.map((hour) => (
                  <button
                    key={hour}
                    type="button"
                    className={`time-option ${hour === hours ? 'selected' : ''}`}
                    onClick={() => handleHourSelect(hour)}
                  >
                    {hour}
                  </button>
                ))}
              </div>
            </div>

            <div className="time-separator">
              <div className="separator-line" />
            </div>

            <div className="time-column">
              <div className="column-header">Minutes</div>
              <div className="time-scroll">
                {minutesArray.map((minute) => (
                  <button
                    key={minute}
                    type="button"
                    className={`time-option ${minute === minutes ? 'selected' : ''}`}
                    onClick={() => handleMinuteSelect(minute)}
                  >
                    {minute}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="time-picker-footer">
            <button
              type="button"
              className="time-picker-done"
              onClick={() => setIsOpen(false)}
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
}