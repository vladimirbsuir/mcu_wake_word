import { useId } from "react";
import { cn } from "@/lib/utils";

export type MetricSeries = {
  label: string;
  color: string;
  values: Array<number | null | undefined>;
};

type MetricChartProps = {
  title: string;
  series: MetricSeries[];
  yMin?: number;
  yMax?: number;
  formatValue?: (value: number) => string;
  emptyLabel?: string;
  className?: string;
};

const WIDTH = 340;
const HEIGHT = 132;
const PADDING_LEFT = 26;
const PADDING_RIGHT = 8;
const PADDING_TOP = 10;
const PADDING_BOTTOM = 20;

export function MetricChart({
  title,
  series,
  yMin = 0,
  yMax,
  formatValue = (value) => value.toFixed(2),
  emptyLabel = "Waiting for training data",
  className,
}: MetricChartProps) {
  const gradientId = useId().replace(/:/g, "");
  const longest = Math.max(...series.map((item) => item.values.length), 0);
  const numericValues = series.flatMap((item) =>
    item.values.filter((value): value is number => typeof value === "number" && Number.isFinite(value)),
  );
  const computedMax = numericValues.length
    ? Math.max(...numericValues) * 1.12
    : yMax ?? 1;
  const resolvedMax = Math.max(yMin + 0.001, yMax ?? computedMax, yMin + 0.001);
  const chartWidth = WIDTH - PADDING_LEFT - PADDING_RIGHT;
  const chartHeight = HEIGHT - PADDING_TOP - PADDING_BOTTOM;
  const stepX = longest > 1 ? chartWidth / (longest - 1) : chartWidth;
  const scaleY = (value: number) =>
    PADDING_TOP +
    chartHeight -
    ((Math.max(yMin, Math.min(resolvedMax, value)) - yMin) / (resolvedMax - yMin)) * chartHeight;
  const latest = series
    .map((item) => {
      const values = item.values.filter(
        (value): value is number => typeof value === "number" && Number.isFinite(value),
      );
      return values.length ? values[values.length - 1] : null;
    })
    .filter((value): value is number => value !== null);

  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className={cn("rounded-lg border border-border/70 bg-background/35 p-3", className)}>
      <div className="mb-2 flex items-start justify-between gap-2">
        <div>
          <div className="label-caps">{title}</div>
          <div className="mt-1 font-mono text-lg leading-none text-foreground">
            {latest.length ? formatValue(latest[latest.length - 1]) : "—"}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          {series.map((item) => (
            <div key={item.label} className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: item.color }} />
              {item.label}
            </div>
          ))}
        </div>
      </div>

      {longest === 0 ? (
        <div className="flex h-[104px] items-center justify-center rounded border border-dashed border-border/60 text-[11px] text-muted-foreground/60">
          {emptyLabel}
        </div>
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="h-[112px] w-full overflow-visible"
          role="img"
          aria-label={`${title} chart`}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={series[0]?.color ?? "#c8f560"} stopOpacity="0.16" />
              <stop offset="100%" stopColor={series[0]?.color ?? "#c8f560"} stopOpacity="0" />
            </linearGradient>
          </defs>
          {gridLines.map((fraction) => {
            const y = PADDING_TOP + chartHeight * fraction;
            const value = resolvedMax - fraction * (resolvedMax - yMin);
            return (
              <g key={fraction}>
                <line
                  x1={PADDING_LEFT}
                  x2={WIDTH - PADDING_RIGHT}
                  y1={y}
                  y2={y}
                  stroke="rgba(255,255,255,0.065)"
                  strokeDasharray="2 4"
                />
                <text
                  x={PADDING_LEFT - 5}
                  y={y + 3}
                  fill="rgba(255,255,255,0.32)"
                  fontSize="7.5"
                  fontFamily="monospace"
                  textAnchor="end"
                >
                  {formatValue(value)}
                </text>
              </g>
            );
          })}
          {longest > 1 && (
            <text
              x={WIDTH - PADDING_RIGHT}
              y={HEIGHT - 4}
              fill="rgba(255,255,255,0.28)"
              fontSize="7.5"
              fontFamily="monospace"
              textAnchor="end"
            >
              epoch {longest}
            </text>
          )}
          {series.map((item, seriesIndex) => {
            const points = item.values
              .map((value, index) => {
                if (typeof value !== "number" || !Number.isFinite(value)) return null;
                return { x: PADDING_LEFT + index * stepX, y: scaleY(value) };
              })
              .filter((point): point is { x: number; y: number } => point !== null);
            if (!points.length) return null;
            const line = points.map((point) => `${point.x},${point.y}`).join(" ");
            const area =
              seriesIndex === 0 && points.length > 1
                ? `${PADDING_LEFT + 0 * stepX},${PADDING_TOP + chartHeight} ${line} ${
                    PADDING_LEFT + (itemsLength(item.values) - 1) * stepX
                  },${PADDING_TOP + chartHeight}`
                : null;
            return (
              <g key={item.label}>
                {area && <polygon points={area} fill={`url(#${gradientId})`} />}
                <polyline
                  points={line}
                  fill="none"
                  stroke={item.color}
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {points.map((point, index) => (
                  <circle
                    key={`${item.label}-${index}`}
                    cx={point.x}
                    cy={point.y}
                    r={index === points.length - 1 ? 2.4 : 1.1}
                    fill={index === points.length - 1 ? item.color : "#111614"}
                    stroke={item.color}
                    strokeWidth="1.1"
                  />
                ))}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

function itemsLength(values: Array<number | null | undefined>) {
  let last = 0;
  values.forEach((value, index) => {
    if (typeof value === "number" && Number.isFinite(value)) last = index;
  });
  return last + 1;
}
