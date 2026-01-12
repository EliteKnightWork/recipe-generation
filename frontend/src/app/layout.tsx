import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Recipe Generator',
  description: 'Generate recipes from your ingredients using AI',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
