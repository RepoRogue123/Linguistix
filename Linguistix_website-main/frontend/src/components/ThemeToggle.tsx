/**
 * SCOPE / PRINT switch.
 *
 * Not labelled light/dark on purpose. These are the two ways this field has
 * always looked at a voice — live on the instrument, or printed on paper — and
 * they change how the spectrogram is drawn, not just the surrounding chrome.
 */

export type Theme = 'scope' | 'print';

interface Props {
  theme: Theme;
  onChange: (theme: Theme) => void;
}

const OPTIONS: { value: Theme; label: string; description: string }[] = [
  { value: 'scope', label: 'Scope', description: 'Live instrument, colour spectrogram' },
  { value: 'print', label: 'Print', description: 'Paper stock, spectrogram as ink density' },
];

export function ThemeToggle({ theme, onChange }: Props) {
  return (
    <div className="theme">
      <span className="eyebrow" id="theme-label">
        Display
      </span>
      <div className="theme__group" role="radiogroup" aria-labelledby="theme-label">
        {OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={theme === option.value}
            title={option.description}
            className={`theme__option${theme === option.value ? ' is-active' : ''}`}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
