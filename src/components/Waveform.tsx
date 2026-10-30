import { useCallback, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

type WaveformProps = {
  levels: number[];
  active: boolean;
  processing?: boolean;
  className?: string;
};

function placeholderLevels(count = 84): number[] {
  return Array.from({ length: count }, (_, index) => {
    const envelope = 0.2 + 0.55 * Math.exp(-Math.pow((index - count * 0.52) / (count * 0.22), 2));
    const texture = 0.52 + 0.32 * Math.sin(index * 1.31) + 0.16 * Math.sin(index * 0.43 + 1.8);
    return Math.max(0.04, envelope * Math.abs(texture));
  });
}

export function Waveform({ levels, active, processing, className }: WaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const levelsRef = useRef(levels);
  levelsRef.current = levels;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== Math.floor(width * dpr) || canvas.height !== Math.floor(height * dpr)) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);

    // Fine time grid gives the recorder a lab-instrument feel.
    context.lineWidth = 1;
    context.strokeStyle = "rgba(255,255,255,0.055)";
    for (let index = 1; index < 8; index += 1) {
      const x = (width / 8) * index;
      context.beginPath();
      context.moveTo(x + 0.5, 16);
      context.lineTo(x + 0.5, height - 16);
      context.stroke();
    }
    context.strokeStyle = "rgba(255,255,255,0.08)";
    context.beginPath();
    context.moveTo(0, height / 2 + 0.5);
    context.lineTo(width, height / 2 + 0.5);
    context.stroke();

    const data = active || processing ? levelsRef.current : placeholderLevels();
    const shown = data.length ? data : placeholderLevels();
    const maxBars = Math.max(48, Math.min(108, Math.floor(width / 7)));
    const sampled: number[] = [];
    for (let index = 0; index < maxBars; index += 1) {
      const sourceIndex = Math.min(
        shown.length - 1,
        Math.floor((index / Math.max(1, maxBars - 1)) * (shown.length - 1)),
      );
      sampled.push(shown[sourceIndex] ?? 0);
    }

    const gap = Math.max(2, width / maxBars / 3.2);
    const barWidth = Math.max(1.6, width / maxBars - gap);
    const total = maxBars * barWidth + (maxBars - 1) * gap;
    const startX = (width - total) / 2;
    const mid = height / 2;
    const maxAmplitude = height * 0.7;

    for (let index = 0; index < sampled.length; index += 1) {
      const level = active || processing ? sampled[index] : sampled[index] * 0.42;
      const amplitude = Math.max(1.4, Math.min(1, level) * maxAmplitude);
      const x = startX + index * (barWidth + gap);
      const color = active
        ? index > sampled.length - 14
          ? "rgba(214, 255, 124, 1)"
          : "rgba(186, 242, 75, 0.92)"
        : processing
          ? "rgba(103, 232, 249, 0.7)"
          : "rgba(152, 164, 155, 0.34)";
      context.fillStyle = color;
      if (active) {
        context.shadowColor = "rgba(186, 242, 75, 0.35)";
        context.shadowBlur = 8;
      } else {
        context.shadowBlur = 0;
      }
      context.beginPath();
      context.roundRect(x, mid - amplitude / 2, barWidth, amplitude, barWidth / 2);
      context.fill();
    }
    context.shadowBlur = 0;

    if (active) {
      const pulse = (Math.sin(performance.now() / 260) + 1) / 2;
      context.fillStyle = `rgba(214, 255, 124, ${0.35 + pulse * 0.45})`;
      context.beginPath();
      context.arc(width - 9, 13, 3, 0, Math.PI * 2);
      context.fill();
    }
  }, [active, processing]);

  useEffect(() => {
    draw();
  }, [levels, draw, processing, active]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => {
    if (!active) return;
    let frame = 0;
    const animate = () => {
      draw();
      frame = window.requestAnimationFrame(animate);
    };
    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, [active, draw]);

  return (
    <canvas
      ref={canvasRef}
      className={cn("h-full w-full", className)}
      aria-hidden="true"
    />
  );
}
