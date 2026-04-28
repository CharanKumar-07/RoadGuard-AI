import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RoadGuard AI | Premium Smart City Monitoring",
  description: "Deep learning-powered accident detection, ANPR, and traffic monitoring for modern cities.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className="font-sans antialiased"
        style={{ fontFamily: "'Inter', ui-sans-serif, system-ui, sans-serif" }}
      >
        {children}
      </body>
    </html>
  );
}
