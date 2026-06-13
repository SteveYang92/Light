const LANG_LABELS: Record<string, string> = {
  zh: "中文",
  en: "EN",
  ja: "JA",
};

function langLabel(code: string): string {
  return LANG_LABELS[code] ?? code.toUpperCase();
}

function ToggleSwitch({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={`relative shrink-0 w-7 h-4 rounded-full transition-colors ${
        checked ? "bg-[#9ca3af]" : "bg-[#333]"
      }`}
    >
      <span
        className={`absolute top-0.5 left-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
          checked ? "translate-x-3" : "translate-x-0"
        }`}
      />
    </button>
  );
}

function Divider() {
  return <span className="w-px h-3.5 bg-[#2a2a2a] shrink-0" aria-hidden="true" />;
}

export default function SubtitleControls({
  languages,
  subLang,
  subEnabled,
  annotationsEnabled,
  hasAnnotations,
  onSubEnabledChange,
  onSubLangChange,
  onAnnotationsChange,
}: {
  languages: string[];
  subLang: string;
  subEnabled: boolean;
  annotationsEnabled: boolean;
  hasAnnotations: boolean;
  onSubEnabledChange: (enabled: boolean) => void;
  onSubLangChange: (lang: string) => void;
  onAnnotationsChange: (enabled: boolean) => void;
}) {
  const showSubtitles = languages.length > 0;
  if (!showSubtitles && !hasAnnotations) return null;

  return (
    <div className="inline-flex items-center gap-2.5 flex-wrap rounded-lg border border-[#1f1f1f] bg-[#141414] px-2.5 py-1.5 text-xs">
      {showSubtitles && (
        <>
          <div className="flex items-center gap-1.5">
            <span className="text-[#888]">字幕</span>
            <ToggleSwitch checked={subEnabled} onChange={onSubEnabledChange} label="字幕" />
          </div>

          {subEnabled && languages.length > 1 && (
            <>
              <Divider />
              <div className="flex items-center gap-0.5" role="group" aria-label="字幕语言">
                {languages.map((lang) => (
                  <button
                    key={lang}
                    type="button"
                    onClick={() => onSubLangChange(lang)}
                    className={`px-1.5 py-0.5 rounded transition-colors ${
                      subLang === lang
                        ? "bg-[#3b82f6]/20 text-[#3b82f6]"
                        : "text-[#6b7280] hover:text-[#e5e5e5]"
                    }`}
                  >
                    {langLabel(lang)}
                  </button>
                ))}
              </div>
            </>
          )}
        </>
      )}

      {hasAnnotations && (
        <>
          {showSubtitles && <Divider />}
          <div className="flex items-center gap-1.5">
            <span className="text-[#888]">注解</span>
            <ToggleSwitch
              checked={annotationsEnabled}
              onChange={onAnnotationsChange}
              label="注解"
            />
          </div>
        </>
      )}
    </div>
  );
}
