
NextJS- React framework
- React Components- user interfaces
- Next.js- additioanl features + optimizations

App router vs Pages router
- App: latest features
- Pages router: original router

# Installation

Recommended- use `create-next-app` which will prompt you

Defaults
- `TypeScript`
- `ESLint`
- `Tailwind`

**Creating directories**
NextJS has file-system routing (routes determine application file structure)

To create a root layout- create a `layout.tsx` and `page.tsx`file in the `app/` folder
and the root layout itself

```jsx
export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
```
And whatever `page.tsx` you want

**public** folder
Store images which you can reference in code 

# Project Structure taxonomy

**Top-Level folders**
- `app`
- `pages
- `public`
- `src`

**Top-Level files**
- `next.config.js`- configuration for nextjs
- `package.json`- dependencies and scripts
- `instrumentation.ts`- Instrumentation file
- `middleware.ts`- NextJS request middleware
- `.env`- env variables
- `.env.local`- local env variables
- `.env.production`- production env variables
- `.env.development`- development env variables
- `.eslintrc.json`- config for ESlint
- `.gitignore`- git files to ignore
- `next-env.d.ts`- ts declaration file for Next.js
- `tsconfig.json`- config file for ts
- `jsconfig.json`- config file for js

## `app` Routing Conventions

**Routing Files**
- `layout`- Layout
- `page`- Page
- `loading`- loading UI
- `not-found`- 
- `error`- 
- `global-error`- 
- `template`- re-rendered layout
- `default` - fallback page
- `route` (only .ts/js)

**Nested routes**
- `folder`- route segment
- `folder/folder`- nested route segment

**Dynamic Routes**
- `[folder]` dynamic route
- `[..folder]` catch-all route segment
- `[[..folder]]` optional catch-all route segment

**Route groups  + private folders**
- `(folder)` Group routes without affecting routing
- `_folder` Opt folder + child segments out of routing

**Parallel / Intercepted Routes**
- `@folder`- named slot
- `(.)folder`- intercept same level
- `(..)folder`- intercept one level above
- `(..)(..)folder`- intercept two levels above
- `(...)folder`- intercept from root

## Metadata

**App Icons**
- `favicon.ico`- favicon file (.ico)
- `icon`- App icon file (.ico/.jpg/.jpeg/.png/.svg)
- `icon`- Generated app icon (.js/.ts.tsx)
- `apple-icon`- apple app icon file (.jpeg/.jpg/.png)
- `apple-icon`- generated apple app icon (.js/.ts/.tsx)

**Open Graph and Twitter images**
- `opengraph-image`- Open Graph image file (jpg/jpeg/png/gif)
- `opengraph-image`- Generated Open Graph image (.js/ts/tsx)
- `twitter-image`- Twitter image file (jpg/jpeg/png/gif)
- `twitter-image`- Generated twitter image (.js/ts/tsx)

**SEO**
- `sitemap.xml`- basic sitemap file
- `sitemap.js/ts`- generated sitemap
- `robot.txt`- robots file
- `robots.js/ts` generated robos file