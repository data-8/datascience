// Registers the Backbone view for the Polymap widget, although for now this is
// a dummy view until we can get everything hooked up properly.
// TODO(sam): Make this actually do things.

require(["widgets/js/widget", "widgets/js/manager"], function(widget, manager) {
    var TestWidgetView = widget.DOMWidgetView.extend({
        render: function() {
            this.$el.text("Hello world!");
        }
    });

    manager.WidgetManager.register_widget_view('TestWidgetView', TestWidgetView);
});
