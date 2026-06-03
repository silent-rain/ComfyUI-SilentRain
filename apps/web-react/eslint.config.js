import js from '@eslint/js';
import prettierConfig from 'eslint-config-prettier';
import prettier from 'eslint-plugin-prettier';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import { defineConfig, globalIgnores } from 'eslint/config';
import globals from 'globals';
import tseslint from 'typescript-eslint';

/**
 * ESLint flat config for Admin Dashboard.
 *
 * Stack: ESLint 9 + TypeScript 5 + React 19 + Vite 6
 */
export default defineConfig([
  // Global ignores
  globalIgnores([
    'dist/**',
    'node_modules/**',
    'build/**',
    'coverage/**',
    'public/**',
    'src/assets/**',
    '*.config.*',
  ]),

  // Base JavaScript recommendations (applied to all files)
  js.configs.recommended,

  // React Hooks (flat config preset)
  reactHooks.configs['recommended-latest'],

  // React Refresh for Vite
  reactRefresh.configs.vite,

  // TypeScript recommended (no type-checking) for all .ts/.tsx files
  ...tseslint.configs.recommended.map(config => ({
    ...config,
    files: ['**/*.{ts,tsx}'],
  })),

  // Type-checked rules ONLY for src/**, since tsconfig.app.json only includes src
  ...tseslint.configs.recommendedTypeChecked.map(config => ({
    ...config,
    files: ['src/**/*.{ts,tsx}'],
  })),

  // Source files (src/**) — full TypeScript + project type information
  {
    files: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser },
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
        ecmaFeatures: { jsx: true },
        project: './tsconfig.app.json',
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: { prettier },
    rules: {
      'prettier/prettier': ['error', { endOfLine: 'auto', semi: true }],

      // TypeScript
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-inferrable-types': 'error',
      '@typescript-eslint/consistent-type-definitions': ['error', 'interface'],
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { prefer: 'type-imports', fixStyle: 'inline-type-imports' },
      ],

      // React Hooks
      'react-hooks/exhaustive-deps': 'warn',

      // General code quality
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-debugger': 'warn',
      'prefer-const': 'error',
      'no-var': 'error',
      'object-shorthand': 'error',
      'prefer-template': 'error',
      'prefer-arrow-callback': 'error',
      semi: ['error', 'always'],
    },
  },

  // Other TypeScript files (mock/**, scripts/**, etc.) — no type-checking
  {
    files: ['**/*.{ts,tsx}'],
    ignores: ['src/**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
      parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module',
      },
    },
    plugins: { prettier },
    rules: {
      'prettier/prettier': ['error', { endOfLine: 'auto', semi: true }],
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-debugger': 'warn',
      semi: ['error', 'always'],
    },
  },

  // Plain JavaScript files (no type-checking)
  {
    files: ['**/*.{js,jsx,mjs,cjs}'],
    extends: [tseslint.configs.disableTypeChecked],
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: { prettier },
    rules: {
      'prettier/prettier': ['error', { endOfLine: 'auto', semi: true }],
      'no-console': ['warn', { allow: ['warn', 'error'] }],
      'no-debugger': 'warn',
      semi: ['error', 'always'],
    },
  },

  // Prettier (must come last to disable conflicting style rules)
  prettierConfig,
]);
