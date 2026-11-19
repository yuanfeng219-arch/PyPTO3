# Deck 本地资源

主入口：`../PyPTO-3.0-deck.html`。

- `Design/p05-*` 至 `Design/p14-*`：从 Design 根目录迁入的 7 个独立叙事页面目录，保留页面、脚本、样式、字体、说明和截图。
- `Design/PyPTO-Compute-Explorer-UI/`：本地产品预览入口及其共享资源副本。
- `vendor/pto-design-system/`：页面使用的设计系统资源副本。

保留 Design/vendor 的相对层级，让嵌入页面及 CSS 的相对引用在 html-deck 内解析。P06 内嵌页通过本地 base 解析资源；独立页面引用主 deck 资源时直接回到 html-deck 根目录。

这些共享资源是 deck 的本地快照；后续更新外部原型时，需要明确同步此处副本。Google Fonts 等网络字体链接仍保留，此目录不承诺完全离线。
